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
