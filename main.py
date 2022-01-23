# Imports
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from numpy.random import randn
from numpy import uint8
from PIL import Image
from nextcord.ext import commands

import nextcord

# Generator
gen_file = open("gen.json", 'r')
gen_json = gen_file.read()
gen_file.close()
G = model_from_json(gen_json)
G.load_weights("gen333.h5")
G.compile(optimizer=Adam(), loss='binary_crossentropy')

# Discriminator
dis_file = open("dis.json", 'r')
dis_json = dis_file.read()
dis_file.close()
D = model_from_json(dis_json)
D.load_weights("dis333.h5")
D.compile(optimizer=Adam(), loss='binary_crossentropy')

# Jazz
GAN = Sequential()
GAN.add(G)
GAN.add(D)
GAN.compile(optimizer=Adam(), loss='binary_crossentropy')


def generate_im():
    ims = G.predict(randn(1, 4096))
    im = (ims[0] + 1.0) / 2.0
    x = Image.fromarray(uint8(im * 255))
    x.save('im.jpg')


# Bot Stuff
bot = commands.Bot(command_prefix="bot ")


@bot.command(name='fake')
async def fake(ctx):
    if ctx.channel.id == 854322964732182528:
        generate_im()
        await ctx.channel.send(file=nextcord.File('im.jpg'))


bot.run('')
