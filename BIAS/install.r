
#install AutoSearch
url <- "https://cran.r-project.org/src/contrib/Archive/AutoSEARCH/AutoSEARCH_1.5.tar.gz"
pkgFile <- "AutoSEARCH_1.5.tar.gz"
download.file(url = url, destfile = pkgFile)
install.packages(c("zoo", "lgarch"))

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)

# Delete package tarball
unlink(pkgFile)

#install PoweR
url <- "https://cran.r-project.org/src/contrib/Archive/PoweR/PoweR_1.0.7.tar.gz"
pkgFile <- "PoweR_1.0.7.tar.gz"
download.file(url = url, destfile = pkgFile)
install.packages(c("parallel", "Rcpp"))

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)

# Delete package tarball
unlink(pkgFile)

#install others
install.packages(c('nortest', 'data.table', 'goftest', 'ddst'))
