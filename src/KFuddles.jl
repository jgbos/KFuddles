module KFuddles

using Knet
using AutoGrad
using Compat
using JLD

export KFuddle, Linear, Conv, DeConv, BatchNorm
export nparams, weights, @sequence
export softmax, leaky_relu, sigm_cross_entropy

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
sigm_cross_entropy(z, y) = max(z, 0) - z .* y + log1p(exp(-abs(z)))

abstract KFuddle
weights(l::KFuddle) = Any[]
weights(ls::Array{KFuddle}) = vcat([weights(l) for l in ls]...)
(layers::Array{KFuddle})(x; kw...) = foldl((x,l)->l(weights(l), x;kw...), x, layers)
(layers::Array{KFuddle})(w, x; kw...) = foldl((x,l)->l(w, x;kw...), x, layers)
nparams(ls::Array{KFuddle}) = sum([nparams(l) for l in ls])
nparams(l::KFuddle) = 2 # TODO: might want to ensure a layer can turn off bias

function save_snapshot(f::String, ls::Array{KFuddle})
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

function load_snapshot(f::String, ls::Array{KFuddle})
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

# Allow unary functions to automatically be KFuddles
immutable Unary{F} <: KFuddle f::F end
Base.convert{F<:Function}(::Type{KFuddle}, f::F) = Unary(f)
Base.convert{F<:Function}(::Type{Unary}, f::F) = Unary{F}(f)
@compat (a::Unary)(w, x; kw...) = a.f(x)
nparams(l::Unary) = 0

Knet.grad(layers::Array{KFuddle}) = Knet.grad(
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

# Helper functions to automatically name flows based on the sequency name

name!(n::KFuddle, s::AbstractString) = (n.name = @compat String(s); n)
function name!(f::KFuddle, name)
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

function name!(flows::Vector{KFuddle}, basename)
  i = 1
  k = 1
  for flow in flows
    if :name in fieldnames(flow)
      name!(flow, string(basename, "_", k))
      k += 1
    end

    if :i_W in fieldnames(flow)
      flow.i_W = 2i - 1
      flow.i_b = 2i
      i += 1
    end
  end
  flows
end

# KFuddle Blocks
kzeros{T}(::Type{T}, s::Vararg{Int}) = KnetArray(zeros(T, s...))
kones{T}(::Type{T}, s::Vararg{Int}) = KnetArray(ones(T, s...))

type Linear <: KFuddle
  name
  W
  b
  i_W
  i_b
end
Linear(W, b) =  name!(Linear("", W, b, 1, 2), "linear")
Linear{T}(W::Array{T}) =  Linear(W, zeros(T, size(W,1),1))
Linear{T}(W::KnetArray{T}) =  Linear(W, kzeros(T, size(W,1),1))
weights(l::Linear) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::Linear)(w, x; kw...) = w[l.i_W]*x .+ w[l.i_b]

type Conv <: KFuddle
  name
  W
  b
  i_W
  i_b
  kw
end
Conv(W, b; kw...) = name!(Conv("", W, b, 1, 2, kw), "conv")
Conv{T}(W::Array{T}; kw...) = Conv(W, zeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)),1); kw...)
Conv{T}(W::KnetArray{T}; kw...) = Conv(W, kzeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)),1); kw...)
weights(l::Conv) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::Conv)(w, x; kw...) = conv4(w[l.i_W], x; l.kw...) .+ w[l.i_b]


type DeConv <: KFuddle
  name
  W
  b
  i_W
  i_b
  kw
end
DeConv(W, b; kw...) = name!(DeConv("", W, b, 1, 2, kw), "deconv")
DeConv{T}(W::Array{T}; kw...) = DeConv(W, zeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)-1),1); kw...)
DeConv{T}(W::KnetArray{T}; kw...) = DeConv(W, kzeros(T, ones(Int, ndims(W)-2)..., size(W,ndims(W)-1),1); kw...)
weights(l::DeConv) = (w=Array(Any, 2); w[1]=l.W; w[2]=l.b; w)
(l::DeConv)(w, x; kw...) = deconv4(w[l.i_W], x; l.kw...) .+ w[l.i_b]


# Batch Normalization Layer
# works both for convolutional and fully connected layers
# mode, 0=>train, 1=>test
type BatchNorm <: KFuddle
  name
  W
  b
  i_W
  i_b
  μ
  σ
  ϵ
end
function BatchNorm(W, b, μ, σ; ϵ=1e-5, kw...)
  bn = BatchNorm("", W, b, 1, 2, μ, σ, ϵ)
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
  return w[l.i_W] .* xhat .+ w[l.i_b]
end


end
