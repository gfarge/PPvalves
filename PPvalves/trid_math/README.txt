The fortran subroutines written in trid_math.f95 are wrapped in python using f2py.
The compiling syntax is in command line : 

`>> f2py -c tmath.f95 -m tmath`

This line compiles the fortran code into a useable f2py module, that can simply
be imported in python using :

`import tmath as tm` (or any other name you'd prefer)


Note: to compile on my system (latest OSX), I have needed to update the `LDFLAGS`
variable: 

`>> LDFLAGS="-shared $LDFLAGS"`
