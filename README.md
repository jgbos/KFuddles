# KFuddles


    Oh what a layer-of-a-layer-of-a-layer-of!
    The sequence is the longest that's ever been heard of.
    So long and so fancy they'd be in a fix
    If they didn't have helpers.  It takes about six
    To tag along hoisting The-KFuddle-dee-Duddle's
    Wonderful sequence out of muddle-dee-puddles.
          --Adapted from Dr. Seuss, On Beyond Zebra

## Usage
```julia
const atype = KnetArray{Float32}

@sequence predict = KFuddle[
    Conv(convert(atype, xavier(5, 5, 1, 20))),
    relu,
    pool,

    Conv(convert(atype, xavier(5, 5, 20, 50))),
    relu,
    pool,
    
    mat,
    
    Linear(convert(atype, xavier(500, 800))),
    relu,
    
    Linear(convert(atype, xavier(10, 500)))
]

 function loss(w,x,ygold)
     ypred = predict(w,x)
     ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
     -sum(ygold .* ynorm) / size(ygold,2)
 end
 
 w = weights(predict)
 o = [Adam() for i in w]
 lossgrad = grad(loss)
 
 function train(w, data; lr=.1, epochs=3, iters=1800)
     for epoch=1:epochs
         for (x,y) in data
             g = lossgradient(w, x, y)             
             update!(w, g, o)
         end
     end
 end
```
