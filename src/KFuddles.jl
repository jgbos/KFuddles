module KFuddles

using Knet
using AutoGrad
using Compat
using JLD

export Fuddle, Linear, Conv, DeConv, BatchNorm, LSTM
export Sequence, ResidualBlock
export nparams, weights, @sequence
export softmax, leaky_relu, sigm_cross_entropy
export receptive_field, transform_size

softmax(x,dim) = let
  m = maximum(x)
  z = exp(x-m)
  z./sum(z, dim)
end

leaky_relu(x, slope=0.2) =  max(x, slope*x)

"""
Sigmoid Cross Entropy Loss
input: un-normalized probabilities (z) and binary class label (y)
"""
sigm_cross_entropy(z, y) = max(z, 0) .- z .* y .+ log1p(exp(-abs(z)))



abstract Fuddle



weights(l::Fuddle) = Any[]
weights(ls::Array{Fuddle}) = vcat([weights(l) for l in ls]...)
#(layers::Array{Fuddle})(x; kw...) = foldl((x,l)->l(weights(l), x;kw...), x, layers)
nparams(ls::Array{Fuddle}) = sum([nparams(l) for l in ls])
nparams(l::Fuddle) = 2 # TODO: might want to ensure a layer can turn off bias

function (layers::Array{Fuddle})(w, x; resblock=false, kw...)
  if !resblock
    return foldl((x,l)->l(w, x;kw...), x, layers)
  else
    Fx = foldl((x,l)->l(w, x;kw...), x, layers[1:end-1])
    return Fx + layers[end](w, x)
  end
end


function save_snapshot(f::String, ls::Array{Fuddle})
  data = Dict{String, Array}()
  for l in ls
    w = weights(l)
    if length(w)==2
      data["$(l.name)_W"] = convert(Array, w[1])
      data["$(l.name)_b"] = convert(Array, w[2])
    end
  end
  save(f, data)
end

function load_snapshot(f::String, ls::Array{Fuddle})
  data = load(f)
  for l in ls
    w = weights(l)
    if length(w)==2
      T = eltype(w[1])
      if isa(w[1], KnetArray)
        atype = KnetArray{T}
      else
        atype = Array{T}
      end
      l.W = convert(atype, data["$(l.name)_W"])
      l.b = convert(atype, data["$(l.name)_b"])
    end
  end
end

# Allow unary functions to automatically be Fuddles
immutable Unary{F} <: Fuddle f::F end
Base.convert{F<:Function}(::Type{Fuddle}, f::F) = Unary(f)
Base.convert{F<:Function}(::Type{Unary}, f::F) = Unary{F}(f)
@compat (a::Unary)(w, x; kw...) = a.f(x)
nparams(l::Unary) = 0

Knet.grad(layers::Array{Fuddle}) = Knet.grad(
  (w, x; kw...)->foldl((x,l)->l(w, x; kw...), x, layers)
)


# Build a sequency of Fuddles
function sequence_(e::Expr)
  if e.head in (:block, :const, :global)
    return Expr(e.head, map(sequence_, e.args)...)
  elseif e.head == :(=) && isa(e.args[1], Symbol)
    return :($(esc(e.args[1])) = name!($(esc(e.args[2])), $(string(e.args[1]))))
  elseif e.head == :line
    return e
  else
    error("unexpected expression: $e")
  end
end

macro sequence(arg...)
  if length(arg) == 2 && (isa(arg[1], AbstractString) || isa(arg[1], Symbol))
    name,node = arg
    return :($(esc(Symbol(name))) = name!($(esc(node)), $(string(name))))
  elseif length(arg) == 1 && isa(arg[1], Expr)
    return sequence_(arg[1])
  else
    error("unexpected arguments to @sequence")
  end
end

# Helper functions to automatically name fuddles based on the sequency name

name!(n::Fuddle, s::AbstractString) = (n.name = @compat String(s); n)
function name!(f::Fuddle, name)
  if :name in fieldnames(f)
    f.name = name
  end

  for fn in fieldnames(f)
    if isdefined(f, fn)
      o = getfield(f, fn)
      name!(o, string(name, "_", fn))
    end
  end

  f
end

function name!(fuddles::Vector{Fuddle}, basename)
  i = 1
  k = 1
  h = 1
  for fuddle in fuddles
    if :name in fieldnames(fuddle)
      name!(fuddle, string(basename, "_", k))
      k += 1
    end

    if :i_W in fieldnames(fuddle)
      fuddle.i_W = i #2i - 1
      # fuddle.i_b = 2i
      i += nparams(fuddle)
    end

    if :i_h in fieldnames(fuddle)
      fuddle.i_h = h
      h += 1
    end
  end
  fuddles
end

kzeros{T}(::Type{T}, s::Vararg{Int}) = KnetArray(zeros(T, s...))
kones{T}(::Type{T}, s::Vararg{Int}) = KnetArray(ones(T, s...))

# ResidualBlock
type ResidualBlock <: Fuddle
  name
  layers::Array{Fuddle}
  i_W
  # i_b
end
ResidualBlock(layers) =  name!(ResidualBlock("", layers, 1), "resblock")
weights(r::ResidualBlock) = vcat([weights(l) for l in r.layers]...)
nparams(r::ResidualBlock) = sum([nparams(l) for l in r.layers])
function (r::ResidualBlock)(w, x; kw...)
  k = 0
  for l in r.layers
    if :i_W in fieldnames(l)
      l.i_W = r.i_W + k
      k += nparams(l)
    end
  end

  Fx = foldl((x,l)->l(w, x;kw...), x, r.layers[1:end-1])
  return Fx + r.layers[end](w, x)
end

# Fuddle Blocks
type Linear <: Fuddle
  name
  W
  b
  i_W
  # i_b
end
Linear(W, b) =  name!(Linear("", W, b, 1), "linear")
Linear{T}(W::Array{T}) =  Linear(W, zeros(T, size(W,1),1))
Linear{T}(W::KnetArray{T}) =  Linear(W, kzeros(T, size(W,1),1))
weights(l::Linear) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::Linear)(w, x; kw...) = w[l.i_W]*x .+ w[l.i_W+1]

transform_size(l::Linear, args...) = size(l.W, 1)


type Conv <: Fuddle
  name
  W
  b
  i_W
  # i_b
  kw
end
Conv(W, b; kw...) = name!(Conv("", W, b, 1, kw), "conv")
Conv{T}(W::Array{T}; kw...) = Conv(W, zeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)),1); kw...)
Conv{T}(W::KnetArray{T}; kw...) = Conv(W, kzeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)),1); kw...)
weights(l::Conv) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::Conv)(w, x; kw...) = conv4(w[l.i_W], x; l.kw...) .+ w[l.i_W+1]

function receptive_field(nin, k, p, s, j=1,r=1,start=0.5)
  nout = transform_size(Conv, nin, k, p, s)
  jout = j*s
  rout = r + (k-1)*j
  sout = start + ((k-1)/2 - p)*j
  return (nout, jout, rout, sout)
end
transform_size(::Type{Conv}, xin, k, p, s, d=1) = floor(Int, (xin + 2p - k - (k-1)*(d-1))/s) + 1
function transform_size(l::Conv, xin)
  p = 0
  s = 1
  for (k, v) in l.kw
    if k==:stride
      s = v
    elseif k==:padding
      p = v
    end
  end
  transform_size(Conv, xin, size(l.W,1), p, s)
end


type DeConv <: Fuddle
  name
  W
  b
  i_W
  # i_b
  kw
end
DeConv(W, b; kw...) = name!(DeConv("", W, b, 1, kw), "deconv")
DeConv{T}(W::Array{T}; kw...) = DeConv(W, zeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)-1),1); kw...)
DeConv{T}(W::KnetArray{T}; kw...) = DeConv(W, kzeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)-1),1); kw...)
weights(l::DeConv) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::DeConv)(w, x; kw...) = deconv4(w[l.i_W], x; l.kw...) .+ w[l.i_W+1]


# Batch Normalization Layer
# works both for convolutional and fully connected layers
# mode, 0=>train, 1=>test
type BatchNorm <: Fuddle
  name
  W
  b
  i_W
  # i_b
  μ
  σ
  ϵ
end
function BatchNorm(W, b, μ, σ; ϵ=1e-5, kw...)
  bn = BatchNorm("", W, b, 1, μ, σ, ϵ)
  name!(bn, "batchnorm")
end

function BatchNorm{T}(W::Array{T}; kw...)
  s = size(W)
  b = zeros(T, s)
  μ = zeros(T, s)
  σ = ones(T, s)
  BatchNorm(W, b, μ, σ; kw...)
end
function BatchNorm{T}(W::KnetArray{T}; ϵ=1e-5, kw...)
  s = size(W)
  b = kzeros(T, s...)
  μ = kzeros(T, s...)
  σ = kones(T, s...)
  BatchNorm(W, b, μ, σ; kw...)
end
weights(l::BatchNorm) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)

function (l::BatchNorm)(w, x; mode=1, kw...)
  mu, sigma = nothing, nothing
  if mode == 0
    d = ndims(x) == 4 ? (1,2,4) : (2,)
    s = prod(Int[size(x,i) for i in d])
    mu = sum(x, d) / s
    x0 = x .- mu
    x1 = x0 .* x0
    sigma = sqrt(l.ϵ + (sum(x1, d)) / s)

    # we need getval in backpropagation
    l.μ = 0.1*AutoGrad.getval(mu) + 0.9*l.μ
    l.σ = 0.1*AutoGrad.getval(sigma) + 0.9*l.σ
  elseif mode == 1
    mu = l.μ
    sigma = l.σ
  end

  xhat = (x.-mu) ./ sigma
  return w[l.i_W] .* xhat .+ w[l.i_W+1]
end

type LSTM <: Fuddle
  name
  W
  b
  i_W
  # i_b
  i_h
end
LSTM(W, b) =  name!(LSTM("", W, b, 1, 1), "lstm")
LSTM{T}(W::Array{T}) =  LSTM(W, zeros(T, size(W,1),1))
LSTM{T}(W::KnetArray{T}) =  LSTM(W, kzeros(T, size(W,1),1))
weights(l::LSTM) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
function (l::LSTM)(w, hcx::NTuple{3, Any}; kw...)
  h, c, x = hcx
  j = l.i_h
  h[j], c[j] = lstm(w[l.i_W], w[l.i_W+1], h[j], c[j], x)
  return (h, c, h[j])
end

function lstm(weight,bias,hidden,cell,input)
  gates   = weight * vcat(hidden, input) .+ bias
  h       = size(hidden,1)
  forget  = sigm(gates[1:h,:])
  ingate  = sigm(gates[1+h:2h,:])
  outgate = sigm(gates[1+2h:3h,:])
  change  = tanh(gates[1+3h:4h,:])
  cell    = cell .* forget + ingate .* change
  hidden  = outgate .* tanh(cell)
  return (hidden,cell)
end


end
