# Audit report

| task | paper | n | OK | OK_NONCONTIGUOUS | RECOVERABLE | UNRECOVERABLE | INDEX_ERROR | OPENA_EMPTY | OPENA_IS_LIST | NO_MATCH |
|---|---|---|---|---|---|---|---|---|---|---|
| q8 | Q7 | 400 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| q9 | Q8 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 400 | 0 |
| q10 | Q9 | 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| q11 | Q10 | 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| q13 | Q12 | 700 | 695 | 0 | 0 | 0 | 1 | 4 | 0 | 0 |
| q14 | Q13 | 300 | 251 | 18 | 7 | 24 | 0 | 0 | 0 | 0 |

## q13

| index | category | closeA | fixed | note |
|---|---|---|---|---|
| 163 | INDEX_ERROR | O3 | O2 | openA='red', options={'O1': 'white', 'O2': 'red', 'O3': 'blue', 'O4': 'silver'} |
| 225 | OPENA_EMPTY | O2 | - | openA='', options={'O1': 'blue', 'O2': 'purple', 'O3': 'white', 'O4': 'turquoise'} |
| 297 | OPENA_EMPTY | O1 | - | openA='', options={'O1': 'purple', 'O2': 'blue', 'O3': 'teal', 'O4': 'white'} |
| 484 | OPENA_EMPTY | O1 | - | openA='', options={'O1': 'brown', 'O2': 'black', 'O3': 'violet', 'O4': 'red'} |
| 688 | OPENA_EMPTY | O2 | - | openA='', options={'O1': 'blue', 'O2': 'navy', 'O3': 'white', 'O4': 'lavender'} |

## q14

| index | category | closeA | fixed | note |
|---|---|---|---|---|
| 1 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 6 | UNRECOVERABLE | O8 | - | openA='Tackling', options={'O1': 'Challenge', 'O2': 'St. tackling', 'O3': 'High leg', 'O4': 'Pushing'} |
| 7 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 8 | UNRECOVERABLE | O8 | - | openA='Challenge', options={'O1': 'Holding', 'O2': 'Tackling', 'O3': 'Dive', 'O4': 'Pushing'} |
| 29 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O1': 'Yes', 'O3': 'No'} |
| 41 | RECOVERABLE | O6 | O4 | openA='Standing tackling', options={'O1': 'Elbowing', 'O2': 'Holding', 'O3': 'Pushing', 'O4': 'St. tackling'} |
| 45 | UNRECOVERABLE | O9 | - | openA='Standing tackling', options={'O1': 'Pushing', 'O2': 'Tackling', 'O3': 'Holding', 'O4': 'Elbowing'} |
| 46 | UNRECOVERABLE | O6 | - | openA='Elbowing', options={'O1': 'High leg', 'O2': 'Tackling', 'O3': 'Holding', 'O4': 'Challenge'} |
| 47 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 53 | UNRECOVERABLE | O7 | - | openA='High leg', options={'O1': 'Challenge', 'O2': 'Holding', 'O3': 'Tackling', 'O4': 'Pushing'} |
| 54 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O1': 'Yes', 'O3': 'No'} |
| 70 | RECOVERABLE | O8 | O2 | openA='Standing tackling', options={'O1': 'Elbowing', 'O2': 'St. tackling', 'O3': 'Challenge', 'O4': 'Dive'} |
| 72 | UNRECOVERABLE | O5 | - | openA='Challenge', options={'O1': 'Pushing', 'O2': 'High leg', 'O3': 'Tackling', 'O4': 'Dive'} |
| 77 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Yellow card or red card', 'O2': 'Yellow card', 'O3': 'Red card', 'O4': 'No card or yellow card'} |
| 83 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Yellow card', 'O2': 'No card or yellow card', 'O3': 'Red card', 'O4': 'Yellow card or red card'} |
| 86 | RECOVERABLE | O9 | O2 | openA='Standing tackling', options={'O1': 'Holding', 'O2': 'St. tackling', 'O3': 'Tackling', 'O4': 'Pushing'} |
| 98 | UNRECOVERABLE | O7 | - | openA='Standing tackling', options={'O1': 'Dive', 'O2': 'Tackling', 'O3': 'Challenge', 'O4': 'Holding'} |
| 107 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O2': 'Yes', 'O3': 'No'} |
| 115 | UNRECOVERABLE | O7 | - | openA='Challenge', options={'O1': 'Dive', 'O2': 'Pushing', 'O3': 'Holding', 'O4': 'St. tackling'} |
| 117 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 121 | UNRECOVERABLE | O7 | - | openA='Challenge', options={'O1': 'Holding', 'O2': 'High leg', 'O3': 'St. tackling', 'O4': 'Pushing'} |
| 128 | UNRECOVERABLE | O8 | - | openA='Tackling', options={'O1': 'Elbowing', 'O2': 'High leg', 'O3': 'Challenge', 'O4': 'St. tackling'} |
| 140 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Red card', 'O2': 'No card or yellow card', 'O3': 'Yellow card', 'O4': 'Yellow card or red card'} |
| 141 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 142 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O1': 'Yes', 'O3': 'No'} |
| 145 | UNRECOVERABLE | O6 | - | openA='Standing tackling', options={'O1': 'High leg', 'O2': 'Pushing', 'O3': 'Tackling', 'O4': 'Holding'} |
| 148 | RECOVERABLE | O7 | O4 | openA='Standing tackling', options={'O1': 'Pushing', 'O2': 'High leg', 'O3': 'Challenge', 'O4': 'St. tackling'} |
| 160 | UNRECOVERABLE | O5 | - | openA='Tackling', options={'O1': 'Challenge', 'O2': 'Elbowing', 'O3': 'Pushing', 'O4': 'Dive'} |
| 166 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 167 | UNRECOVERABLE | O8 | - | openA='Tackling', options={'O1': 'Holding', 'O2': 'Challenge', 'O3': 'St. tackling', 'O4': 'Elbowing'} |
| 168 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'No card or yellow card', 'O2': 'Yellow card or red card', 'O3': 'Yellow card', 'O4': 'Red card'} |
| 169 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 176 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 180 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Yellow card or red card', 'O2': 'Red card', 'O3': 'Yellow card', 'O4': 'No card or yellow card'} |
| 191 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Red card', 'O2': 'Yellow card or red card', 'O3': 'No card or yellow card', 'O4': 'Yellow card'} |
| 203 | RECOVERABLE | O9 | O2 | openA='Standing tackling', options={'O1': 'Pushing', 'O2': 'St. tackling', 'O3': 'Elbowing', 'O4': 'High leg'} |
| 208 | RECOVERABLE | O5 | O3 | openA='Standing tackling', options={'O1': 'Tackling', 'O2': 'High leg', 'O3': 'St. tackling', 'O4': 'Challenge'} |
| 217 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 221 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 223 | UNRECOVERABLE | O5 | - | openA='Standing tackling', options={'O1': 'Dive', 'O2': 'Pushing', 'O3': 'Challenge', 'O4': 'Tackling'} |
| 242 | OK_NONCONTIGUOUS | O2 | - | openA='No', options={'O2': 'No', 'O3': 'Yes'} |
| 255 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'No card or yellow card', 'O2': 'Yellow card or red card', 'O3': 'Red card', 'O4': 'Yellow card'} |
| 258 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O2': 'Yes', 'O3': 'No'} |
| 260 | OK_NONCONTIGUOUS | O1 | - | openA='No', options={'O1': 'No', 'O3': 'Yes'} |
| 266 | UNRECOVERABLE | O9 | - | openA='Standing tackling', options={'O1': 'High leg', 'O2': 'Challenge', 'O3': 'Tackling', 'O4': 'Holding'} |
| 274 | OK_NONCONTIGUOUS | O3 | - | openA='No', options={'O1': 'Yes', 'O3': 'No'} |
| 280 | RECOVERABLE | O6 | O3 | openA='Standing tackling', options={'O1': 'High leg', 'O2': 'Tackling', 'O3': 'St. tackling', 'O4': 'Holding'} |
| 281 | UNRECOVERABLE | O9 | - | openA='Standing tackling', options={'O1': 'Dive', 'O2': 'Challenge', 'O3': 'Holding', 'O4': 'Elbowing'} |
| 285 | UNRECOVERABLE | O5 | - | openA='No card', options={'O1': 'Yellow card', 'O2': 'No card or yellow card', 'O3': 'Yellow card or red card', 'O4': 'Red card'} |
