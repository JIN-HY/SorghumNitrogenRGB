import pandas as pd 
import numpy as np  
import sys
import statistics

plants = ["J" + str(x)[1:] for x in range(1001,1037)]
machines = ['Vis_SV_0', 'Vis_SV_36', 'Vis_SV_72','Vis_SV_90', 'Vis_SV_108','Vis_SV_144','Vis_SV_216', 'Vis_SV_252','Vis_SV_288','Vis_SV_324']

presults = {}
results = {}

trait = sys.argv[1]

for plant in plants:
    presults[plant] = {}
    results[plant] = {}
    for machine in machines:
        df = pd.read_csv(f"{plant}_{machine}_rgbtraits.csv")
        for date in df.date:
            if date not in presults[plant]:
                presults[plant][date] = {machine:float(df[df.date == date]['PixelCount'])}
                results[plant][date] = {machine:float(df[df.date == date][trait])}
            else:
                presults[plant][date].update({machine:float(df[df.date == date]['PixelCount'])})
                results[plant][date].update({machine:float(df[df.date == date][trait])})
            

for plant in plants:
    for date in presults[plant]:
        presults[plant][date] = sorted(presults[plant][date].items(),key=(lambda i: i[1]))
        print(presults[plant][date])
        mainview = presults[plant][date][-1][0]
        mainview2 = presults[plant][date][-2][0]
        results[plant][date] = statistics.mean([results[plant][date][mainview], results[plant][date][mainview2]])

resultsdf = pd.DataFrame.from_dict(results, orient='index')
resultsdf.to_csv(f"{trait}4implant.csv", index = None)
