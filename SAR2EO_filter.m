clear all;
close all;

imgdir = 'D:\dataset\PBVS-translation\test\sar2eo\SAR\';
savedir = 'D:\dataset\PBVS-translation\test\sar2eo\PPB\';
% imgpath = 'Gotcha0.png';
files = dir(imgdir);



for i =3:length(files)
    disp([num2str(i-2),':', files(i,1).name])

    %imgpath = 'Gotcha15.png';
    imgpath = [imgdir, files(i,1).name];
    savepath = [savedir, files(i,1).name];
    
    if(exist(savepath,'file'))
        continue
    end
    oriim = imread(imgpath);
    % figure;
    % imshow(oriim);
    im = double(oriim) / 255.0;
    % im SAR image
    % P sim win, default:3
    % W search win, default:10
    % h smooth, try:0.5 3.5
    P = 3;
    W = 10;
    h = 0.5;
    if sum(sum(im>0))==0
        continue
    end
    im_fil=FAST_PPB(im,P,W,h);

    im_out = (im_fil-min(min(im_fil)))/(max(max(im_fil))-min(min(im_fil)))*255;
    im_out = uint8(im_out);
    imwrite(im_out, savepath);
%     figure;
%     imshow(im_out);
    
end
