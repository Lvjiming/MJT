function wave=Wave_despeckle(I)

[c, s] = wavedec2(I, 2, 'db4');
sigma = median(abs(c)) / 0.6745;
thr = sigma * sqrt(2 * log(length(c)));
c_t = wthresh(c, 's', thr);
wave = waverec2(c_t, s, 'db4');
end