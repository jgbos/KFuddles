# Allow unary functions to automatically be Fuddles
immutable Unary{F} <: Fuddle f::F end
Base.convert{F<:Function}(::Type{Fuddle}, f::F) = Unary(f)
Base.convert{F<:Function}(::Type{Unary}, f::F) = Unary{F}(f)
@compat (a::Unary)(w, x; kw...) = a.f(x)
nparams(l::Unary) = 0


softmax(x,dim) = let
  m = maximum(x)
  z = exp(x-m)
  z./sum(z, dim)
end

leaky_relu(x, slope=0.2) =  max(x, slope*x)
