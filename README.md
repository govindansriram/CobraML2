# CobraML

-----------------

## TIPS

- sudo cpupower frequency-set --governor performance
  - stabilizes cpu on intel

## TODO:

## FEB 17:
- Add non loop unrolled GEMV double kernel
- test on vtune 
- add more loop unrolling if it helps
- look at double implemtation in egyptian paper

## FEB 22:
undo changes
transition code to use padding
create versions for all other datatypes
take break

make same code for neon but possibly use inline asm

## FEB 23:
- Update double to use fmad
- add padding alligment
- take break (do either threading lib or GC)
- add versions for other datatypes
- add mat mul and other mathemetical operators
- add gradient descent
- add gpu support
