import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
import statsmodels.api as sm

from collections import OrderedDict

colormap = {'France': '#318CE7', 'Mexico': '#000000', 'Germany': '#00A76D',
            'Canada': '#FF0000', 'China': '#FFDE00', 'Japan': '#FA8072',
            'US': '#3C3B6E', #'#002663' #'#002C77'
            'IE': '#00CCFF', 'FireFox': '#D64203',#'#CF5714',
            'Chrome': '#4CB749'}

data = pd.read_csv('data_output.csv', parse_dates=[0])#, index_col=[0])
pd.pivot_table(data, values='retained', rows='country')

countries = data.groupby(['country']).size()
countries.sort(ascending=False)

browsers = data.groupby(['browser']).size()
browsers.sort(ascending=False)

group_by = 'country'#'browser
labels = data.groupby([group_by]).size()
labels.sort(ascending=False)
labels = labels.keys()#[:-2]# #leave Germany and Mexico out #[1:] # plot 'US' separately
# use aggfunc=size for counts/sales (values can be any)
df = pd.pivot_table(data, values=values, rows=[group_by, 'date'])#, aggfunc=size)

def plot_ts(df, values='tutorial'):
    fig = plt.figure()
    for label in labels:
        ts = df[label]
        color = colormap[label]
        ts.plot(c=color, ls='--', label='') #, legend=False)
        pd.ewma(ts, 5).plot(c=color, label=label)
    remove_duplicate_labels(3, 0.5)
    ylabel(values)

def plot_tutorial_vs_retained(df, labels):
    fig = plt.figure()
    for label in labels:
        x = df.tutorial[label]
        y = df.retained[label]
        color = colormap[label]
        plt.plot(x, y, 'o', c=color, label=label)
        result = pd.ols(y=y, x=x)
        plt.plot(x, result.y_fitted, c=color, lw=2)#, label=label)
    leg = plt.legend(loc='best', fancybox=True)
    return fig

def ggplot_tutorial_vs_retained(df, labels, se=True):
    p = ggplot(aes(x='tutorial', y='retained'), data=df)
    for label in labels:
        x = df.tutorial[label]
        y = df.retained[label]
        color = colormap[label]
        p = p + geom_point(x=x, y=y, color=color, label=label)
        p = p + stat_smooth(x=x, y=y, color=color, method='lm', se=se)
    print(p)
    leg = plt.legend(loc='best', fancybox=True)
    return p

def remove_duplicate_labels(ncol=1, alpha=1):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg = legend(by_label.values(), by_label.keys(), ncol=ncol, fancybox=True)
    leg.get_frame().set_alpha(alpha)

def fix_hist_labels_colors_ticks(ax):
    patches, labels = ax.get_legend_handles_labels()
    labels=['Chrome', 'FireFox', 'IE']
    for i,label in enumerate(labels):
        for p in patches[0][i::3]:
            p.set_color(colormap[label])

    xticklabels = ax.get_xticklabels()
    #t.set_rotation(0)

def logit_regression(df):
    dummy_country = pd.get_dummies(df.country)
    dummy_browser = pd.get_dummies(df.browser)
    cols_to_keep = ['retained', 'tutorial']
    # NOTE: avoid LinAlgError: Singular matrix - take one dummy out of each
    data = df[cols_to_keep].join(dummy_country.ix[:, 'China':]).join(dummy_browser.ix[:, 'FireFox':])
    data['intercept'] = 1.0
    train_cols = data.columns[1:]
    logit = sm.Logit(data['retained'], data[train_cols])
    result = logit.fit()
    print result.summary()
    return result
