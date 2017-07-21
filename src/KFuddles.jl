module KFuddles

using Knet
using AutoGrad
using Compat
using JLD

export Fuddle, Linear, Conv, DeConv, BatchNorm, LSTM
export Sequence, ResidualBlock, SkipResidualBlock
export nparams, weights, @sequence
export softmax, leaky_relu, sigm_cross_entropy
export receptive_field, transform_size


"""
Sigmoid Cross Entropy Loss
input: un-normalized probabilities (z) and binary class label (y)
"""
sigm_cross_entropy(z, y) = max(z, 0) .- z .* y .+ log1p(exp(-abs(z)))

kzeros{T}(::Type{T}, s::Vararg{Int}) = KnetArray(zeros(T, s...))
kones{T}(::Type{T}, s::Vararg{Int}) = KnetArray(ones(T, s...))


abstract Fuddle
weights(l::Fuddle) = Any[]
weights(ls::Array{Fuddle}) = vcat([weights(l) for l in ls]...)
#(layers::Array{Fuddle})(x; kw...) = foldl((x,l)->l(weights(l), x;kw...), x, layers)
nparams(ls::Array{Fuddle}) = sum([nparams(l) for l in ls])
nparams(l::Fuddle) = 2 # TODO: might want to ensure a layer can turn off bias
windex(l::Fuddle, i) = (l.i_W=i; l)
(layers::Array{Fuddle})(w, x; kw...) = foldl((x,l)->l(w, x;kw...), x, layers)


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
      windex(fuddle, i)
      # fuddle.i_W = i #2i - 1
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

include("unary.jl")
include("base.jl")
include("blocks.jl")
include("recurrent.jl")


end
