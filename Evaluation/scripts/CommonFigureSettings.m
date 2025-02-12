% Figure setting for pubblication quality
%
% Author: Alessandro Fornasier

% Options: Image dim
Unit = 'inches';
XSize = 10;
YSize = 6;
    
% Options: fonts
TitleFontSize = 18;
LegendFontSize = 18;
AxisTickFontSize = 16;
% TitleFontSize = 22;
% LegendFontSize = 22;
% AxisTickFontSize = 18;

% Options: whether/how to print figures
savefigimages.save                         = true;
savefigimages.figOptions.Units             = Unit;
savefigimages.figOptions.PaperUnits        = Unit;
savefigimages.figOptions.Position          = [0 0 XSize YSize];
savefigimages.figOptions.PaperPosition     = [0 0 XSize YSize];
savefigimages.figOptions.PaperSize         = [XSize YSize];
savefigimages.figOptions.PaperPositionMode = "auto";

savefigimages.figOptions.Color    = 'white';
% savefigimages.figOptions.Renderer = 'opengl';
savefigimages.figOptions.Renderer = 'painters';
savefigimages.printOptions        = {'-m4','-svg','-png' '-nocrop'};

% Options: Show or not the image
savefigimages.Visibility          = 'on';
savefigimages.closeAfterSave      = false;
set(groot, 'DefaultFigureVisible', savefigimages.Visibility);

% Options: Save
plots = true;
save_plots = plots && false;
print_rmse = false;

% Options: Colors
colorR = '#EE3377';
colorG = '#009988';
colorB = '#0077BB';
colorGC = '#FFB7B2';
colorBD = '#6B7B8E';