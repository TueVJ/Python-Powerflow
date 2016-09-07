import pandas as pd
import defaults

from Optimize_Market import Optimize_Market

# Load wind scenarios
wind_sc = pd.read_csv(defaults.wind_scenario_file, index_col=0)
wind_cap = pd.Series.from_csv(defaults.wind_capacity_file, index_col=0)

# Wind scenarios depend on scenario s, time t and node n; they are more convenient for a panel, which has 3 indices.
tempgb = wind_sc.groupby('minor')
# Relative production
wind_sc_rel = pd.Panel({g: tempgb.get_group(g).drop('minor', axis=1) for g in tempgb.groups}).swapaxes(0, 1).swapaxes(1, 2)
# Production with capacity layout
wind_sc = pd.Panel({g: wind_cap[g]*tempgb.get_group(g).drop('minor', axis=1) for g in tempgb.groups}).swapaxes(0, 1).swapaxes(1, 2)

# Dimensions: 10 (items) x 48 (major_axis) x 14 (minor_axis)
# Items axis: s0 to s9
# Major_axis axis: t0 to t47
# Minor_axis axis: n1 to n9

# Load demand signal
load_signal = pd.read_csv(defaults.load_file, index_col=0)

# Dimensions: 14 (columns) x 48 (index)
# Columns axis: n1 to n9
# index axis: t0 to t47

m = Optimize_Market(wind_sc, load_signal)

m.optimize() # Optimal value 5.53e+05

# Plots wind scenarios

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.ion()
plt.figure()
wind_sc.sum(axis=2).plot(c='k')
plt.gca().legend_.remove()
plt.ylabel('Wind production [MWh]')
plt.title('Wind production per scenario')
sns.despine(offset=8)
plt.tight_layout()

plt.figure()
ax = plt.axes()

# Plot generator production
gprod = pd.Panel(
    [[[m.variables.gprod[s, g, t].x for g in m.data.generators] for t in m.data.taus] for s in m.data.scenarios],
    items=m.data.scenarios, major_axis=m.data.taus, minor_axis=m.data.generators)
loadshed = pd.DataFrame(
    [[sum(m.variables.loadshed[s, n, t].x for n in m.data.nodeorder) for s in m.data.scenarios] for t in m.data.taus],
    index=m.data.taus, columns=m.data.scenarios)
# Load served is upper bound on load, less the amount shed
loadserved = pd.DataFrame(
    [[sum(m.variables.loadshed[s, n, t].ub - m.variables.loadshed[s, n, t].x for n in m.data.nodeorder) for s in m.data.scenarios] for t in m.data.taus],
    index=m.data.taus, columns=m.data.scenarios)
for s, df in gprod.iteritems():
    df.plot(label=str(s), ax=ax)
ax.legend_.remove()
