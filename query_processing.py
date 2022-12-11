from numpy.random import choice, rand
from numpy import around

from loader import load_image
import sys

if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
        data_df = load_image(path)

        # randomly generate image stance
        data_df['stance'] = choice(['PRO', 'CON'], size=data_df.shape[0])
        data_df['score'] = around(rand(data_df.shape[0])*100, 2)
        data_df['method'] = 'random'
        data_df['rank'] = choice(range(1,10), size=data_df.shape[0])

        # arrange output file. see touche's page for submission format
        result = data_df[['topic', 'stance', 'image_id', 'rank', 'score', 'method']].sort_values(by=['topic', 'image_id'])
        result.to_csv('result.tsv', sep='\t', index=False, header=False)

    else:
        print("please provide path to the image dataset")
    

