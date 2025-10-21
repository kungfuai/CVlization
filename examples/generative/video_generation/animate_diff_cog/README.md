Adapted from https://github.com/lucataco/cog-animatediff

Install `cog`:

```
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

Then in this directory,

```
cog build -t animatediff
cog predict -i prompt="masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, outdoors, falling petals, white hair, brown eyes"
```