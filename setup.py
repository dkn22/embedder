from distutils.core import setup

setup(name='embedder',
      version='0.1',
      description='Embed categorical variables via neural networks',
      author='Dat Nguyen',
      author_email='dat.nguyen@cantab.net',
      url='https://github.com/dkn22/embedder',
      packages=['embedder',],
      license='BSD',
      classifiers=[
          'Environment :: MacOS X',
          'Environment :: Win32 (MS Windows)',
          'Environment :: X11 Applications',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries :: Python Modules']
      )
