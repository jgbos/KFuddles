# ResidualBlock
type ResidualBlock <: Fuddle
  name
  layers::Array{Fuddle}
  i_W
end
ResidualBlock(layers) =  name!(ResidualBlock("", layers, 1), "resblock")
weights(r::ResidualBlock) = vcat([weights(l) for l in r.layers]...)
nparams(r::ResidualBlock) = sum([nparams(l) for l in r.layers])
function windex(r::ResidualBlock, i)
  r.i_W = i
  k = 0
  for l in r.layers
    if :i_W in fieldnames(l)
      l.i_W = r.i_W + k
      k += nparams(l)
    end
  end
end

function (r::ResidualBlock)(w, x; kw...)
  Fx = r.layers[1:end-1](w, x; kw...)
  return Fx + r.layers[end](w, x; kw...)
end

# Skip Residual
type SkipResidualBlock <: Fuddle
  name
  layers::Array{Fuddle}
  skip
  i_W
end
SkipResidualBlock(layers, skip=length(layers)) =  name!(SkipResidualBlock("", layers, skip, 1), "skipresblock")
weights(r::SkipResidualBlock) = vcat([weights(l) for l in r.layers]...)
nparams(r::SkipResidualBlock) = sum([nparams(l) for l in r.layers])
function windex(r::SkipResidualBlock, i)
  r.i_W = i
  k = 0
  for (i, l) in enumerate(r.layers)
    if :i_W in fieldnames(l)
      l.i_W = r.i_W + k
      k += nparams(l)
    end
  end
end

function (r::SkipResidualBlock)(w, x, y; kw...)
  for (i, l) in enumerate(r.layers)
    if i == r.skip
      y = l(w, y; kw...)
      x = x + y
    else
      x = l(w, x; kw...)
    end
  end
  return x, y
end
