"""Script for plotting data.

On error bars:
https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/

On setting labels:
https://www.delftstack.com/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/

https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib

https://stackoverflow.com/questions/28288383/python-matplotlib-plot-x-axis-with-first-x-axis-value-labeled-as-1-instead-of-0

https://stackoverflow.com/questions/22481854/plot-mean-and-standard-deviation

Font family:
https://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pandas as pd

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser('plotting statst')

    parser.add_argument('read_path', help='csv path', type=str)

    parser.add_argument(
        'write_path', help='plot path (with valid pic. file ext).')

    parser.add_argument('x_label', help='x-label for graph.', type=str)

    parser.add_argument('y_label', help='y-label for graph.', type=str)

    parser.add_argument('title', help='title for graph', type=str)

    parser.add_argument(
        '--plot_type',
        choices=['bar', 'errorbar'],
        help='type of graph to plot. (default: errorbar)',
        default='errorbar')

    parser.add_argument(
        '--x_size',
        help='font size for x label',
        default=16,
        type=int)

    parser.add_argument(
        '--xtick_label_size',
        help='font size for x-tick labels',
        default=16,
        type=int)

    parser.add_argument(
        '--xtick_label_step',
        help='the increment for the x-tick labels. (default: 1)',
        type=int,
        default=1)

    parser.add_argument(
        '--y_size',
        help='font size for y label',
        default=16,
        type=int)

    parser.add_argument(
        '--ytick_label_size',
        help='font size for y-tick labels',
        default=16,
        type=int)

    parser.add_argument(
        '--title_size',
        help='font size for y label',
        default=16,
        type=int)

    parser.add_argument(
        '--markersize',
        help='size of marker for errbar chart only. (default: 9)',
        default=12,
        type=int)

    parser.add_argument(
        '--figsize',
        help='width by length (space separated). (default: None)',
        default=None,
        nargs='+',
        type=int)

    parser.add_argument(
        '--font_family',
        help='set font family globally. (default: sans-serif)',
        type=str,
        default='sans-serif')

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.read_path)

    # Plotting elements
    # first ele is the outer loop var name
    xtick_labels = data[data.columns[0]]
    x_pos = np.arange(1, len(xtick_labels)+1)
    errs = data['std_error']
    means = data['mean']

    # Set global style
    plt.rcParams['font.family'] = args.font_family

    # Plotting
    fig, ax = plt.subplots(figsize=args.figsize)

    # Bar plot
    if args.plot_type == 'bar':
        ax.bar(x_pos, means,
               yerr=errs,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)

    elif args.plot_type == 'errorbar':
        ax.errorbar(x_pos, means, yerr=errs, fmt='o',
                    markersize=args.markersize)

    # Set the x label
    ax.set_xlabel(args.x_label, fontsize=args.x_size)

    # # Use to set ticks
    ax.set_xticks(x_pos)

    # Used to set fontsize
    ax.set_xticklabels(xtick_labels,
                       fontsize=args.xtick_label_size)

    # Used to set spacing of tick labels
    ax.xaxis.set_major_locator(
        matplotlib.ticker.MultipleLocator(args.xtick_label_step))

    # Set y label
    ax.set_ylabel(args.y_label, fontsize=args.y_size)

    # Set ytick label size
    plt.setp(ax.get_yticklabels(), fontsize=args.ytick_label_size)

    # Set chart title
    ax.set_title(args.title, size=args.title_size)

    # Put the grid on
    ax.yaxis.grid(True)

    # Save the figure
    fig.savefig(args.write_path, bbox_inches='tight')
