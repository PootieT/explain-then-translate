FROM ubuntu:22.04
RUN apt-get update -yqq && apt-get install -yqq curl build-essential python3-pip python3-tqdm
RUN apt-get install racket -yqq
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update
RUN apt-get install -yqq \
    default-jdk-headless \
    golang-go \
    php-cli \
    ruby \
    lua5.3 \
    r-base \
    rustc \
    scala

RUN apt-get install -yqq libtest-deep-perl 
RUN apt-get install -yqq wget 

# JS/TS
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - 
RUN apt-get install -y nodejs
RUN npm install -g typescript

# Dlang
RUN wget https://netcologne.dl.sourceforge.net/project/d-apt/files/d-apt.list -O /etc/apt/sources.list.d/d-apt.list
RUN apt-get update --allow-insecure-repositories
RUN apt-get -y --allow-unauthenticated install --reinstall d-apt-keyring
RUN apt-get update && apt-get install -yqq dmd-compiler dub

# C#
RUN apt install gnupg ca-certificates
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list
RUN apt update
RUN apt install -yqq mono-devel

# Post-processing

# Julia
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz | tar xz
ENV PATH="/julia-1.8.2/bin:${PATH}"
# Swift
RUN curl https://download.swift.org/swift-5.7-release/ubuntu2204/swift-5.7-RELEASE/swift-5.7-RELEASE-ubuntu22.04.tar.gz | tar xz
ENV PATH="/swift-5.7-RELEASE-ubuntu22.04/usr/bin:${PATH}"
# Javatuples
RUN mkdir /usr/multiple && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar
# Luaunit
RUN apt-get update -yqq && apt-get install -yqq lua-unit
# R - add common libraries
RUN wget https://cran.r-project.org/src/contrib/sets_1.0-24.tar.gz
RUN R CMD INSTALL sets_1.0-24.tar.gz
RUN wget https://cran.r-project.org/src/contrib/stringi_1.7.12.tar.gz
RUN R CMD INSTALL stringi_1.7.12.tar.gz
RUN wget https://cran.r-project.org/src/contrib/R.methodsS3_1.8.2.tar.gz \
         https://cran.r-project.org/src/contrib/R.oo_1.25.0.tar.gz \
         https://cran.r-project.org/src/contrib/R.utils_2.12.2.tar.gz
RUN R CMD INSTALL R.methodsS3_1.8.2.tar.gz R.oo_1.25.0.tar.gz R.utils_2.12.2.tar.gz
RUN wget https://cran.r-project.org/src/contrib/cli_3.6.1.tar.gz \
         https://cran.r-project.org/src/contrib/glue_1.6.2.tar.gz \
         https://cran.r-project.org/src/contrib/lifecycle_1.0.4.tar.gz \
         https://cran.r-project.org/src/contrib/magrittr_2.0.3.tar.gz \
         https://cran.r-project.org/src/contrib/rlang_1.1.2.tar.gz \
         https://cran.r-project.org/src/contrib/vctrs_0.6.4.tar.gz \
         https://cran.r-project.org/src/contrib/stringr_1.5.0.tar.gz
RUN R CMD INSTALL cli_3.6.1.tar.gz glue_1.6.2.tar.gz rlang_1.1.2.tar.gz lifecycle_1.0.4.tar.gz magrittr_2.0.3.tar.gz vctrs_0.6.4.tar.gz stringr_1.5.0.tar.gz
RUN wget https://cran.r-project.org/src/contrib/hash_2.2.6.3.tar.gz
RUN R CMD INSTALL hash_2.2.6.3.tar.gz

COPY src /code
WORKDIR /code
ENTRYPOINT ["python3", "main.py"]
