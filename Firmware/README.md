## IMPORTANT: Instructions are intended for Mac user only

## Setting up Mbed:

1. Install package based on OS: https://os.mbed.com/docs/mbed-os/v5.14/quick-start/offline-with-mbed-cli.html

2. run the tool

3. Command

   ```bash
   $ cd Proj__StewartPlatform/Firmware/Nucleo_main
   $ mbed deploy
   $ mbed compile --target NUCLEO_F401RE --toolchain GCC_ARM
   ```

## Install GCC_ARM

1. Install GCC ARM with homebrew: https://github.com/osx-cross/homebrew-arm

2. and we can build locally:
```bash
   $ cd Proj__StewartPlatform/Firmware/Nucleo_main
   $ make
```

3. Remember to manually update Makefile for new libs.

