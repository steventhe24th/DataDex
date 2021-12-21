from distutils.core import setup
setup(
  name = 'Datadex',         # How you named your package folder (MyLib)
  packages = ['Datadex'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Data Science Framework',   # Give a short description about your library
  author = 'steventhe24th',                   # Type in your name
  author_email = 'stevenivanderjunaidi@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/steventhe24th/DataDex',   # Provide either the link to your github or to your website
  download_url = '',    # I explain this later on
  keywords = ['Framework', 'Experiment', 'Pipeline'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
        'pandas',
        'numpy',
        'scikit-learn',
        'keras_tuner',
        'spacy',
        'tensorflow-text'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)