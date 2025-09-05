# Friction Log - Minhash Deduplication on Common Crawl

- Date: Sep 5, 2025
- Author: Everett Kleven
- Size: M
- Persona: LLM Dataset Curation Data Scientist
- [Notebook](/workload/minhash_dedupe_common_crawl.ipynb)
- [Python Script](/workload/structured_outputs_workload.py)
  
## Summary


## Context

## Friction Points
The [unfiltered notebook](daft-structured-outputs/friction/full_notebook_unfiltered.ipynb) captures my experience end-to-end building the 

### Minor - But necessary - The hot pink throughout the docs is abrasive and hurts my eyes. 
I've got nothing against the recent brand redesign, but the hot pink needs to be toned down a bit. The AI age tends to encourage maximalism in color choice and aesthetic appeal, however the particular tone of daft's primary color is a bit to harsh. I figured I might as well mention this as a part of an actual issue I ran into that is a bit more practical. 

I was trying to take advantage of `daft.from_glob_path` to grab all the file paths from common crawl with a matching wildcard url signature and then ran into this funny usage pattern where I realized I couldn't download the warc files with `df.url.download()`. This might sound obvious, but even as a daft power user I forgot that I can just wildcard `daft.read_warc()`. The `from_glob_path` reader screams "use me to find files with wildcards" and so thats where my mind went. 

Malcolm and I had a short discussion about this in the slack channel talking about imprvoing documentation clarity around wildcard usage with `IO` ops, which I think is fair. This is a more useful friction point to talk about once we begin to consider how often engineers end up manually partioning files in s3 prefixes for certain formats. 

The average raw data pipeline prioritizes ingestion to storage in a raw gzipped format. You risk losing data if your ingestion workload has to successfully preprocess before it lands in tables (whether that be in a data lake or data warehouse). When I was at lucid all of our pipelines synced to s3 and were prefixed by a few diferent keys that we knew where the main filters our end users had to specify to retrieve their data. 

For vehicle logs, we had these resumable http endpoints that sinked to VIN/day/ECU prefixes. These logs were processed lazily since the consumption rate of logs was so low. 

My point here is that its super common for data engineers to impelement specific prefix structures on raw data in s3 which requires to be read with non-trivial wildcard addresses. This is a super common and practical use case for demonstrating `IO` reads/writes with wildcards. Especially for raw data. 

I'd know that Daft can support wildcards if I just looked at the second section of the front page of the guide. The only thing is my eyes breeze right past the demonstrated usage patterns because the pink formatting on the code is so intense. My eyers literally avert themselves from being acausted. (ok dramatic, but that was my subconcious reaction)

### Pre-processing WARC HTML Payloads 

I'm a bit hesitant to even include this section because I'm not sure my headaches were so much a matter of daft not supplying helpful primatives, and more so  of parsing the warc_content payload. A part from learning about the payload format itself, which is a byte string comprised of both the HTTP header as well as the payload, there were a couple strategies that I tried with daft that didn't end up working the way I expected. 

### Large Star and Short Star Debugging Headache

By far, the most frustrating part of this workload was trying to debug and understand the graph operations inside the Connected components section. The entire workload algorithm was new to me, but this section really threw me through a loop. I kept running into a strange error relating to 
 
## Raw Take

First thing I want to say is that for anyone who is new to minhash dedupe, the algorithm itself has a lot of new vocabulary. Deduplication is an important workload for pretraining....

### Daft Pain points in the developing the multimodal structured outputs workload

#### Preprocessing



## Conclusion

My goal in developing this workload was to not only uncover friction points in scaling daft multimodal inference, but to also provide a genuinely useful piece of code. AI engineers are constantly evaluating new models, and rarely do you see examples that are this end to end.

I'd like to extend the workload to compare performance across model variants, or explore how different sampling parameters impact performance. There seems to be a new frontier open source model released every week nowadays, and I only expect that trend to continue.

I think there are a lot of pain points that I have uncovered that are easily adderessed with a new llm_generate api signature. Throughout this exercise I aimed to focus on the problems instead of trying to provide an all-encompassing solution. I hope you found this Friction log helpful and maybe learned a thing or two a long the way.