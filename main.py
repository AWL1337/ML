from parse.parceSite import parce_site
from export.toArff import save_to_arff
from export.toTsv import save_to_tsv

data = parce_site()

save_to_tsv(data)
save_to_arff(data)
