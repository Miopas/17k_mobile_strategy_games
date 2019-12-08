import requests
import shutil
import pandas as pd
import sys

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    icons_path = './icons/'

    df = pd.read_csv(infile)
    icon_urls = df['Icon URL']
    ratings = df['Average User Rating']
    IDs = df['ID']

    with open(outfile, 'w') as fw:
        for i, url in enumerate(icon_urls):
            if (ratings[i] != ratings[i]): # NaN
                continue
            path = icons_path + 'icon_{}.png'.format(IDs[i])
            with open(path, 'wb') as f:
                try:
                    r = requests.get(url, stream=True)
                    if r.status_code == 200:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                except:
                    sys.stderr.write('error id: {}, url:{}\n'.format(IDs[i], url));
                f.close()
            fw.write('icon_{}\t{}\n'.format(IDs[i], ratings[i]));
    fw.close()

