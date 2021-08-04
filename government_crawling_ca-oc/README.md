# The Catalan - Occitan Government Crawling

The Catalan Catalan - Occitan Government Crawling is a corpus for Machine Translation composed of 503 Catalan - Occitan aligned sentences.

The corpus has been obtained by crawling the .gencat domain and subdomains, belonging to the Catalan Government during September and October 2020. We use the [CorpusCleaner](https://github.com/TeMU-BSC/corpus-cleaner-acl) pipeline to process the WARC files obtained from the crawling. This allows us to maintain the metadata and retrieve the original url per each document. We extract the content of the same URLs in both languages and align them at document level using [vecalign](https://github.com/thompsonb/vecalign). The final dataset was obtained by manually reviewing 1,237 automatically aligned sentences.

We license the actual packaging of this data under a CC0 1.0 Universal License.
