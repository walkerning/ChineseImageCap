# -*- coding: utf-8 -*-

import sys
base = 9000
readf = open(sys.argv[1], "r")
writef = open(sys.argv[2], "w")
for i, line in enumerate(readf):
    writef.write(str(i+base) + " " + "".join(line.split()) + "\n")
