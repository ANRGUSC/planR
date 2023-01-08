#!/bin/bash
## run program X times consecutively
#seq 5 | xargs -Iz python3 main.py

#AlphaArray=(0.1 0.15 0.2 0.25)
#for i in "${AlphaArray[@]}"
#do
#  python3 main.py "$i" &> "out/$i" &
#done
#wait

NAlphaArray=(0.21 0.22 0.23)
for j in "${NAlphaArray[@]}"
do
  python3 main.py "$j" &> "out/$j" &
done
