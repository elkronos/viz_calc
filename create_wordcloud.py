from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Function to create a word cloud with further enhancements
def create_wordcloud(text, max_words=200, width=800, height=400, background_color='white', colormap='viridis', save_to_file=False, file_name='wordcloud.png', mask=None, interactive=False):
    if interactive:
        max_words = int(input("Please enter the maximum number of words: "))
        width = int(input("Please enter the width: "))
        height = int(input("Please enter the height: "))
        background_color = input("Please enter the background color: ")
        colormap = input("Please enter the colormap: ")
        save_to_file = bool(int(input("Save to file? Enter 1 for Yes, 0 for No: ")))
        if save_to_file:
            file_name = input("Please enter the file name (with extension): ")

    wordcloud = WordCloud(max_words=max_words, width=width, height=height, background_color=background_color, colormap=colormap, contour_color='blue', contour_width=2, mask=mask).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    if save_to_file:
        wordcloud.to_file(file_name)
        print(f'Word cloud saved to: {file_name}')

    plt.show()

# Synthetic data about cats
synthetic_data = '''
Cats are known for their playful and curious nature. They are one of the most popular pets worldwide, loved for their grace and elegance. Cats are independent animals, often preferring to explore their surroundings at their own pace. They have sharp retractable claws and keen senses, which make them excellent hunters.

In the wild, cats are solitary predators, often stalking their prey stealthily before making a quick and decisive strike. Domestic cats retain many of these natural instincts, which can sometimes be observed when they play with toys or chase after laser pointers. Despite their predatory nature, cats have also been known to form close bonds with humans and other animals, including dogs, rabbits, and even birds.

Cats have a strong territorial instinct and they often mark their territory by leaving scent markings or by scratching surfaces. This behavior is natural and helps them establish their presence and deter intruders. Cats communicate using a variety of vocalizations, including meows, purrs, and hisses. They also use body language to express themselves, such as arching their back and puffing up to appear larger when threatened.

Cats require a balanced diet to maintain their health and vitality. It is recommended to provide them with high-quality cat food that meets their nutritional needs. It's also important to provide them with fresh water and to keep their living environment clean and stimulating.

Cats are known to have a calming effect on humans and spending time with a cat can help reduce stress and improve mood. Their purring is not only soothing, but it has also been found to have a therapeutic effect, promoting healing and reducing pain and inflammation.

In summary, cats are fascinating creatures with a rich and complex behavior. They are loved and adored by many, and continue to captivate the hearts of cat enthusiasts all over the world.
'''

# Example usage
create_wordcloud(synthetic_data, save_to_file=True)
