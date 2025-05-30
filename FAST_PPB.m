%% fast PPB
% 输入：
    % im SAR image
    % P sim win, default:3
    % W search win, default:10
    % h smooth, try:0.5 3.5
% out：
    % im_fil
% ref：
% [1]Deledalle, C-A; Denis, L.; Tupin, F., "Iterative Weighted Maximum Likelihood Denoising With Probabilistic Patch-Based Weights," Image Processing, IEEE Transactions on , vol.18, no.12, pp.2661,2672, Dec. 2009
% [2]胡开洋,耿伯英.  基于预筛选的改进的SAR图像PPB去斑[J]. 计算机工程与设计. 2013(03)
% [3]Darbon, J.; Cunha, A.; Chan, T.F.; Osher, S.; Jensen, G.J., "Fast nonlocal filtering applied to electron cryomicroscopy," Biomedical Imaging: From Nano to Macro, 2008. ISBI 2008. 5th IEEE International Symposium on , vol., no., pp.1331,1334, 14-17 May 2008
function im_fil=FAST_PPB(im,P,W,h)
%% initialization
im(im==0)=min(im(im>0));
[Height,Width]=size(im);

x=1:Height;
y=1:Width;
xP=1:Height+2*P+1;
yP=1:Width+2*P+1;

O=zeros(size(im));
M=zeros(size(im));
Z=zeros(size(im));
im_pad=padarray(im,[W+P+1,W+P+1],'symmetric');
%% loop

for dx=-W:W

    for dy=-W:W
        
        if dx == 0 && dy == 0
            continue;
        end
        
        a=im_pad(xP+W,yP+W);
        b=im_pad(xP+W+dx,yP+W+dy);
        SD=log(a./b+b./a);
        ISD=cumsum(cumsum(SD,1),2);
        SSD=ISD(x,y)+ISD(x+2*P+1,y+2*P+1)-ISD(x,y+2*P+1)-ISD(x+2*P+1,y);
        w=exp(-SSD/h);
        v=im_pad(x+W+P+1+dx,y+W+P+1+dy);
        O=O+w.*v.^2;
        M=max(M,w);
        Z=Z+w;
    end
end

O=O+M.*im_pad(x+W+P+1,y+W+P+1).^2;
O=O./(Z+M);
im_fil=O(x,y).^0.5;
