import matplotlib as mpl

mpl.rcParams['backend'] = 'TkAgg'
f = mpl.font_manager.findSystemFonts(fontpaths=['/home/sandrews/extra_fonts/'])
[mpl.font_manager.fontManager.addfont(font_file) for font_file in f]
mpl.rcParams['font.family'] = 'Helvetica'
