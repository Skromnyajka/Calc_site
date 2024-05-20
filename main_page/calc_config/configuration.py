import re
from math import lcm

def multiple_replace(string, replacements):
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    return pattern.sub(lambda m: replacements[m.group(0)], string)

def sum_fract(ln, ld):
    print(ln, ld)
    nok = lcm(*ld)
    print(nok)
    ln_new = []
    for i in range(len(ln)):
        ln_new.append(nok//ld[i]*ln[i])
    return sum(ln_new), nok