from inspect import signature

from matilda.stages.cloister import Cloister

a = signature(Cloister.run)
print(a.parameters)

for p in a.parameters:
    print(a.parameters[p].annotation)
