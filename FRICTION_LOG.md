# Friction Log - Minhash Deduplication on Common Crawl

- Date: Sep 5, 2025
- Author: Everett Kleven
- Size: M
- Persona: LLM Dataset Curation Data Scientist
- [Notebook](/workload/minhash_dedupe_common_crawl.ipynb)
- [Python Script](/workload/structured_outputs_workload.py)
  
## Summary

Overall, I had a pretty positive experience using daft for this workload. I definitely felt more empowered than burdened across several key areas. While my aim for any daft workload is to use as many daft primatives as possible, it felt easy to write UDFs to preprocess the warc payloads and parsing html, but even that felt like a breeze. I then quickly found the text normalization and minhash expressions which made the next two steps of the pipeline fully trivial. I didn't need to know what I was doing, I just ran the transformation and everything worked! Once I got LSH banding I asked GPT-5 to build the next steps and it achieved it within a few tries. I got to this step within the first day which made me feel a little bit like superman, especially since this was the first time I had ever been introduced to the Minhash Deduplication Workload.

I had the opposite experience with Connected Components. Unfortunately, the reference pipeline I was referencing couldn't run without errors and I ended up having to ditch the approach. I go into full detail on my various headaches in the sections below, but eventually I ended up successfully impelementing connected components from referencing the original *Connected Components in MapReduce and Beyond* paper. Throughout this time, I wasn't fighting daft's semantics more so than I was debugging my implementation of the algorithm. In fact I found daft to be quite empowering at this stage as well. The wrapped parenthesis pattern is particularly helpful for interactive debugging and helped me isolate errors quickly.

The most important take away in my experience was my debugging experience trying to get the connected components section to work. This was where I ran into the most pain. The algorithm is non-trivial to implement and represents a real opportunity to simplify the lives of users. The minhash and normalization expressions worked flawlessly the first time, and I found their api's intuitive and trivial to implement. Getting access to s3 worked the first time and setting the `IOConfig` was trivial. I didn't remember how to set the default io config globally and had a hard time finding the 

## Context

This weeks workload was focused on Minhash Deduplication spanning the following steps: 

- AWS Authentication
- Loading and Preprocessing Common Crawl WARC Tables
- Text Normalization
- MinHash Calculation
- LSH Banding
- Connected Components
- Merge Results


It was around this time that I got the reference script from the team and began adjusting my approach to match. You can see how my unfiltered notebook kind of abrubtly ends around connected components. I was able to run everything up until the large star algorithm which threw an exception for a small bug in `col("u").list.map(...)` operation that needed to be adjusted to referencethe wrong column, but since I was still naive to what I was really changing I then transitioned to interactive debugging in a script. Little did I know just how many rabbit holes I was about to go down.

For the next several days I debugged trying the existing approach. At first I was just trying to get the connected components section to run, referencing the provided script as well as the pyspark implementation. I was eventually able to get the workload running, but it yielded no duplicates. Then I was able to yield duplicates, but the number of rows grew in size, instead of reducing. Finally I was seeing the results I was expecting but I was unable to validate them against igraph. The results weren't close. I knew somewhere along the way I had improperly implemented the Large and Small star trasnformations and was having a difficult time identifying where I had gone wrong.

Most of these headaches were due to my unfamliarity with the theory, and less so with daft. daft had all the core primatives that I needed. I just struggles to translate the theory into practice. Eventualy I had to start from first principles and break down the operations step by step.

At this point, the work had bled into the weekend and I was desperate for an answer. It was late into the morning when I decided to leverage an AI agent to help me achieve parity with iGraph. The [explanation from GPT-5](/friction/connected_components_reasoning.md) details the solution it found in the moment.

After careful review, one of the sections it added was just fixing an error I had in my Small Star operation and the other turned out to be genuinely necessary. This was the 


In the end, it was this breakthrough that helped me complete the worklaod and validate my results.

I tested multiple orders of maginitude of documents locally 


## Friction Points
The [unfiltered notebook](daft-minhash-dedupe/friction/UNFILTERED_minhash_dedupe_common_crawl.ipynb) captures my experience end-to-end building the workload up until I began debugging connected components. I

### 1. No usage examples for setting the default io_config 

The `set_planning_config(default_io_config=IO_CONFIG)` is really only documented in the API and is probably the most common way people will authenticate with cloud providers. Searching ioconfig in the docs only yields the `IOConfig` class and usage examples only implement it as a temporary credential.  I'm not sure how often users will ever set io_config more than once, so I'd argue most examples should set the default. 

### 1. Forgot about Wildcard usage patterns and the hot pink accent color

For some reason when I attempted to load WARC files using I decided to grab all of the file paths using `daft.from_glob_path()` (probably to get a deterministic list of warc files to read) which I then naively tried to load with `url.download()`. Unfortunately the operation hangs indefinitely in a jupyer notebook which means I have to restart the kernel and try something new. Having something that fails fast here with solid error messages would be nice. 

Naturally this brings up the question of why I didn't try `daft.read_warc(uri)` right away. In the moment I was hoping to read a specific file to work with the same data every time. In fact, in all of my scripts I am using a non deterministic wildcard pattern and end up loading different data every time. This turned out to be great for developing this workload robustly, but end

### 2. Wildcard Usage Pattern Confusion with `url.download()`

**Problem**: When attempting to load WARC files using `daft.from_glob_path()` followed by `url.download()`, the operation hangs indefinitely. The intuitive pattern of using `df.with_column("warc", col("path").url.download())` fails for S3 paths. This is definitely user error, but since a jupyter notebook cell hangs until it crashes the kernel it feels more like a bug.

**Root Cause**: The `url.download()` doesn't play nicely with tables 

**Solution Discovered**: Use `daft.read_warc()` directly with wildcard patterns instead of the two-step approach. 

I was trying to take advantage of `daft.from_glob_path` to grab all the file paths from common crawl with a matching wildcard url signature and then ran into this funny usage pattern where I realized I couldn't download the warc files with `df.url.download()`. This might sound obvious, but even as a daft power user I forgot that I can just wildcard `daft.read_warc()`. The `from_glob_path` reader screams "use me to find files with wildcards" and so thats where my mind went.

Malcolm and I had a short discussion about this in the slack channel talking about imprvoing documentation clarity around wildcard usage with `IO` ops, which I think is fair. This is a more useful friction point to talk about once we begin to consider how often engineers end up manually partioning files in s3 prefixes for certain formats.

The average raw data pipeline prioritizes ingestion to storage in a raw gzipped format. You risk losing data if your ingestion workload has to successfully preprocess before it lands in tables (whether that be in a data lake or data warehouse). When I was at lucid all of our pipelines synced to s3 and were prefixed by a few diferent keys that we knew where the main filters our end users had to specify to retrieve their data.

For vehicle logs, we had these resumable http endpoints that sinked to VIN/day/ECU prefixes. Since tmost logs are never looked at, we just compressed the raw payloads and left it in s3 to be processed on demand.

My point here is that its super common for data engineers to impelement specific prefix structures on raw data in s3 which requires to be read with non-trivial wildcard addresses. This is a super common and practical use case for demonstrating `IO` reads/writes with wildcards. Especially for raw data.

I'd know that daft can support wildcards if I just looked at the second section of the front page of the guide. The only thing is my eyes breeze right past the demonstrated usage patterns because the pink formatting on the code is so intense. My eyers will just avert subconciously. 

Obviously this is minor. 

### Pre-processing WARC HTML Payloads

I'm a bit hesitant to even include this section because I'm not sure my headaches were so much a matter of daft not supplying helpful primatives, and more so learning how to parse the warc_content payload. A part from learning about the payload format itself, which is a byte string comprised of both the HTTP header as well as the payload, there were a couple strategies that I tried with daft that didn't end up working the way I expected.

### Slow-brain - can't figure out list construction for minimum over multiple columns. 
I'm running into this funny moment where I want to take the minimum between a known variable and a calculated row-wise vairable and can't think of how to do it without a row-wise udf. In the band explosion, if you use the naive B, then you risk trying to create entries for minhashes that dont exist. I ran into this when I transitioned the pipeline from full document deduplication to block level deduplication where an individual block may have less than `B` tokens in it's contents.

For now I'll just stick with a udf since its intutive, but since I have to derive a `min` argumnet from an existing column, the operation turns into more steps than I would have liked. This pain point is fundamentally due to not being able to easily append or build lists. 

Coming back to this, I ended up finding the `list_` constructor, but ended up adopting a two step approach and keeping the udf. 

### Large Star and Short Star Debugging Headache

By far, the most frustrating part of this workload was trying to debug and understand the graph operations inside the Connected components section. The minhash dedupe workload was new to me, but this section really threw me through a loop. I kept running into a strange errors in the `list.map` operation, and then when I finally figured out how to fix each of the expressions to not throw exceptions I kept getting zero deduplication. I knew something was wrong, and then ventured for the next 5 days to figure out a way to get deduplication to work.

When I initially announced success, I had done so prematurely. I had only been able to get the large star operation to run without errors. Little did I know how much longer it would take until I turned it into the correct version, using the [*Connected Components in MapReduce and Beyond*] paper as a reference. Eventually I reached my limit of technical understanding and leaned on multiple independent sessions with gpt-5 and grok 4 to validate my implementation of the algorithms were canonically correct. Grok-4 was invaluable here. GPT-5 was great and all, but it kept throwing me through this debugging loop adding little operations that I couldn't know weren't a part of the paper. Grok 4 shut that down real fast. With the paper attached it tore my code to pieces and provided me a key insight on handling null values inside the large and small star operations that ended up helping me succeed.
 
## Raw Take

First thing I want to say is that for anyone who is new to minhash dedupe, the algorithm itself has a lot of new vocabulary...

i am coming back to this raw take after having spent all weekend trying to get connected components to achieve parity with igraph... and i finally got it. i am exhausted... but i learned a ton. can't even capitalize my letters. 

This is a seminal workload in distributed computing and I think the hardest part of all of this was just not really knowing what I was fiddling with. I had to work with GPT-5 and Grok-4 in order to figure out where my bugs where and they were super nuanced.

I honestly can't even begin to explain just how many iterations I tried, but in the end I had to diverge from the provided reference just to get both large and small star working. Then even after I did that, remapped the integer labels back to the warc-record block ids, i was still seeing small differences between igraph results and my final assignments. They were usually on the order of like 0.001 % of the edges, but since I could work up the scale, I eventually had gpt-5 autonomously identify a solution. Having igraph as a reference was huge. NetworkX is the canonical reference for python, but its probably 20x slower. (iGraph's core is written in C)

You'll find an additional step in my implementation of connected components called `global_min_label_propagation` which has a detailed docstring and comments on how it works. If you look at the operations closely, the `group_by` and neighbor minimization appear extremely similar to the small star operation, but with an important difference. The `edges.join(labels, left_on="v", right_on="u", how="left")` is not the same thing as a canonicalized or symmetricized edge list. The other difference is that this is repeating this same operation until the label pairs are equal. It does end up taking multiple iterations as I've observed, so its not a one-op silver-bullet.

This global label reduction helps cleanup the differences I was seeing in my igraph parity validation tests. I personally have no idea what the precedent is for an operation like this. Its definitely not in the original paper, but it works and leverages dataframe ops so I'm not sure I care.

Considering the amount of frustration and pain I had in debugging the correctness of the algorithm, I'm just glad I have something that works and scales. I mean, what else do we want? This is probably my first real encounter with artificial intelligence yielding real material emergent results and its pretty spectacular. Naturally I'd love to see some more testing around this to identify if it might become a bottleneck at the trillion document level, but seriously... I'm just happy to be done and have something that works. 

#### The algo was the hardest part.

I really can't emphasize enough just how much this week's focus was on the algorithm than daft itself. With a few minor exceptions, daft had pretty much everything I needed to build this solution, it was just a matter of getting everything working and validated. If anything, daft made it super easy to debug anywhere I needed to. I mentioned this on the slack channel but I really like this wrapped parentheses pattern for aggregating lazy transformations.

I can just drop a breakpoint anywhere in the transformation list, add a `.show()` and inspect the data. I can also quickly add select or where filters to quickly visualize just the data I care about, and the pretty table I get in the terminal is just nice to look at. I know we kind of take it for granted at this point, but when I was working with pandas for converting the data to the igraph constructor, I was surprised at how opaque the prints were.
