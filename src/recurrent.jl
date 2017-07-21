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
