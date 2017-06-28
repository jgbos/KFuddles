# KFuddles


    Oh what a layer-of-a-layer-of-a-layer-of!
    The sequence is the longest that's ever been heard of.
    So long and so fancy they'd be in a fix
    If they didn't have helpers.  It takes about six
    To tag along hoisting The-KFuddle-dee-Duddle's
    Wonderful sequence out of muddle-dee-puddles.
          --Adapted from Dr. Seuss, On Beyond Zebra

## Usage

First define a sequence:

```julia
@sequence predict = KFuddle[
    Conv(xavier(5, 5, 1, 20)), # Initializes bias vector with zeros
    relu, # unary functions can be used in place
    pool,

    Conv(xavier(5, 5, 20, 50)),
    relu,
    pool,
    
    mat,
    
    Linear(xavier(500, 800)),
    relu,
    
    Linear(xavier(10, 500))
]
```

The variable `predict` is a function and weight container.  To get the weights,

```julia
w = weights(predict)
```
Execute `predict` for a given feature matrix,

```julia
p = predict(w, x)
```

Additionally, the parameter layers are automatically labeled so that you can easily save and load snapshots, 

```julia
save_snapshot("weights.jld", predict)
load_snapshot("weights.jld", predict)
```

