# Friction Log - Minhash Deduplication on Common Crawl

- Date: Sep 5, 2025
- Author: Everett Kleven
- Size: L
- Persona: LLM Dataset Curation Data Scientist
- [Notebook](/workload/minhash_dedupe_common_crawl.ipynb)
- [Python Script](/workload/minhash_dedupe.py)
- [Results](/workload/results.txt)  


## Context

This weeks workload was focused on Minhash Deduplication spanning the following steps: 

- Loading Common Crawl WARC Tables
- Preprocessing HTML Payloads into text
- Text Normalization
- MinHash Calculation
- LSH Banding
- Connected Components
- Merging Results


## Summary

Thanks to daft, writing this minhash dedupe pipeline was a positive experience. I always aim to use as daft primatives where possible and I felt consistently empowered to do so. I only used a UDF to preprocess the warc payloads, parse html, and build a block index, of which the block index was the only udf I wish I could have avoided. The others felt necessary. 

I also found the text normalization and minhash expressions to be fully trivial to implement which was just really cool. With no built-in LSH Banding function, I spent a little more time verifying my work, but it was fricitonless.

My experience with Connected Components, as unfortunate as it was had more to do with implementing the theory than any headaches in composition. While the reference pipeline was an invaluable resource for me to understand the scope and intended shape of the workload, I was unable to get it to run as delivered. Eventually I ended up impelementing connected components from first priciples referencing the original *Connected Components in MapReduce and Beyond* paper. Throughout this time, I wasn't fighting daft's semantics more so than I was debugging my implementation of the algorithm. In fact, I found daft to be quite empowering at this stage as well. The wrapped parenthesis pattern is particularly helpful for interactive debugging and helped me isolate errors quickly.

I ran the pipeline at a variety of scales on the Native Runner, with promising results. It does appear that eager materialization improves performance significantly following preprocessing, and lsh banding. This deserves more investigation, but I think the reality is that with the number of explodes and groub_by's in this pipeline, it just makes sense to break it up.


## Grease Points (The Good)





## Friction Points (The Bad)

The [unfiltered notebook](daft-minhash-dedupe/friction/UNFILTERED_minhash_dedupe_common_crawl.ipynb) captures my experience end-to-end building the workload up until I began debugging connected components. I

### 1. No usage examples for setting the default io_config

The `set_planning_config(default_io_config=IO_CONFIG)` is really only documented in the API and is probably the most common way people will authenticate with cloud providers. Searching ioconfig in the docs only yields the `IOConfig` class and usage examples only implement it as a temporary credential. I'm not sure how often users will ever set io_config more than once, so I'd argue most examples should set the default.

### 2. List Handling Headaches

Having to come up with a UDF for get_block_idx - a comment on list handling
Two things. There were a couple of moments where I intuitively wanted to mutatute lists but coudln't think of / find  an easy translation in Daft.

#### CASE 1: Appending a scalar column to a list column and taking the minimum of the result

This operation occurs in both the Large/Small Star Reduction after canonicalization/symmetrization. While the algorithm is specialized, I think this use case common. The use case is basically that, everytime you groub_by with an agg_list you want to perform some sort of compound operation on the list in combination with another column. It's one thing to mutate the list with a literal, but since we have to reference a column or previous expression in order to calculate the minimum, it takes mental gymnastics to translate what you would normally do in python to a daft expression.

In our Large/Small Star operations, this scenario occurs after we:

`group_by(col("u")).agg_list("v")`  

and we want to calculate the minimum label of both `col("u")` (int) and `col("neigh")` (list).

```python
...
# Reduce ⟨u; N ⊆ Γ(u)⟩:
df = df.with_column("neigh", df.groupby("u").agg_list("v"))

# IMPLEMENTED
# m = argminv∈Γ+(u) lv - multi-step process
df_multi_step = df.with_column("m", col("nbrs").list.min())
df_multi_step = df_multi_step.with_column(
    "m", 
    col("m").is_null().if_else( # Also ended up having to handle nulls
        col("u"),
        (col("u") < col("m")).if_else(col("u"), col("m"))
    )
)

# DESIRED
# m = argminv∈Γ+(u) lv - with list append (intuitive)
df_list_append = df.with_column("m", col("neigh").list.append(col("u")).list.min())

# Emit stage stays the same 
...
```

Instead of building a new list and taking the minimum of the result, I ended up breaking the operation into stages with something that looked a lot less intuitive. In fact, upon inspection of my current implementation, it's not entirely obvious what I'm doing. I also end up having to handle null values where, if I was appending an empty list with `col("u")` I wouldn't need to.

We also don't want to use the `list_` constructor since we would have to unpack `col("neigh")` list just to make a new one that also included `col("u")`. The number of operations is really only minimized with a `list.append` approach.

@srilman ran into the same thing here and has since added the list.append expression which is demonstrated here.

#### CASE 2: Deterministic Block Index Construction

Initially my minhash pipeline only used a monotonically increasing id, implemented very early in the pipeline, but when I recognized that my id's couldn't be deterministically generated across sessions, I realized I needed more robust approach. Looking back now, even this falls short. I should have just built this secondary index as a record-id, tag + hash  of the contents or something. Regardless, this was what I ended up with.

In the moment, it was just difficult for me to imagine constructing a sibling list with variable sizes without a python function. I decided it wasn't worth the effort and just implemented the UDF.

```python
# See /friction/list_construction.py for repro

@daft.func()
def get_block_idx(blocks: list[str]) -> list[int]: # <-- this
    """Generate integer indices for each element in the input list of blocks."""
    return list(range(len(blocks)))

df_text = (
    df_html
    .with_column("blocks", extract_blocks(col("content_raw")))
    .with_column("block_idx", get_block_idx(col("blocks"))) # <-- here
    .explode("blocks", "block_idx")
    .where(col("blocks") != "")
    .where(col("blocks").not_null())
    .with_column(index_col, col("WARC-Record-ID")+ "-" + col("block_idx"))
    .with_column(content_col, col("blocks"))
    .select(
        "WARC-Record-ID",
        index_col,
        content_col,
    )
)
```

### 2. Evangelizing the Wildcard Usage Pattern

Ok so bare with me on this, I know its a bit of an essay but the conversation is a bit nuanced. I need to touch on:

- Common Crawl HTTP access only allows explicit paths (no wildcards)
- url.download() doesn't work for WARC Files (user wishful thinking)
- User first instinct is to build a list of paths to read
- Most engineers who work with raw data organize with folders/prefixes.

When attempting to load **WARC** files using `daft.from_glob_path()` followed by `url.download()` the operation hangs.

#### *"Why not just use daft.read_warc()?"*  

For Common Crawl, you can either access the warc files through HTTP or, if you have AWS creds, through s3. The problem is, if you try to use wildcard to read multiple files over the HTTP connection, daft will throw an error for invalid wildcard usage or "no files found".

I tried a variety of patterns including `*` and `**` in as many permutations as I could think of, but I was only ever able to read data over HTTP IFF I provided with an explicit path to a specific WARC file. This led me to try to discover files using `from_glob_path` which implies the usage of `url.download()`.

From my experience, with accessing remote data from websites, this tends to be the case for security reasons, so I didn't associate it with an issue with Daft, so much as Common Crawls data access policies. If you want to read multiple files from Common Crawl over HTTP, you have need to extract the paths from the index files.
The alternative to `df.with_column("warc", col("path").url.download())` would be to serialize `paths = df["path"].to_pydict()` and I am severly allergic to this.

There definitely was a part of me that wanted to test the fantasy that ALL of the IO methods were somehow automatically integrated into `url.download()` and could just `read_warc()` and unpack the results. LOL!

Eventually I gave up and set up AWS creds from my own cloud acct.

Malcolm and I ended up having a short discussion about this in the slack channel and theres a nuanced points we touched on.

#### *"do our docs suck?"*

No, no they don't. [On the second section of the opening page](https://docs.getdaft.io/en/stable/quickstart/#read-from-a-data-source), it details multiple demonstrations of wilcard usage in reading data. I knew daft supported wildcards, but since I was running into the issues with HTTP I forgot and ended trying to frakenstein my approach.

If we expand the conversation to reflect how often engineers end up manually organizing files in prefixes/folders, the importance of wildcard usage patterns solidifies. The average raw data pipeline prioritizes ingestion to storage in a raw gzipped format. You risk losing data if your ingestion workload has to successfully preprocess before it lands in tables (whether that be in a data lake or data warehouse).

When I was at Lucid Motors, all of our pipelines sinked straight from our load balancer to s3. We organized the payloads by prefixing writes with date/VIN prefixes.

Its super common for engineers to implement specific prefix structures on raw data in s3. We all know this. That's why Daft supports arbitrary glob patterns. But do we showcase that feature consistently enough? Does every IO expression come with usage examples detailing the canonical wildcard patterns?

If I am just checking out daft, I'll read the front page, but its just as likely I will jump straight to the API or user guide for my specific thing and see how to that there. Thats what I need, not the "generic way of reading data with daft".

Devils advocate would say, "I'd know that daft can support wildcards if I just looked at the second section of the front page of the guide". This is true, but the usage patterns demonstrate wildcards *implicitly* through demonstration. It's not *explicitly spelled* out in a sentence.

*"Engineers are smart enough to recognize the usage patterns"*.

Yes. Absolutley. Theres just one catch with this. I'm not sure if anyone else feels similar, but my eyes breeze right past the demonstrated usage patterns because the pink formatting on the code is so intense. When I went to look for the usage patterns, my retinas told me to breeze past them. Maybe this is just a dark mode thing, but it is worth mentioning.

## Raw Take (The Ugly)

First thing I want to say is that for anyone who is new to minhash dedupe, the algorithm itself has a lot of new vocabulary...

Up until connected components, things went pretty smoothly. Having never implemented it before, I was pretty encouraged by my ability to progress through to the minhash stage without many hiccups. LSH Banding was a little complicated, but with a quick prompt to GPT-5, I had something that worked really quickly.

### Connected Components

By far, the most frustrating part of this workload was trying to debug and understand the graph operations inside the Connected Components section. I kept running into a strange errors in the `list.map` operation, and then when I finally figured out how to fix it, something else would throw.

It was around this time that I got the reference script from the team and began adjusting my approach to match. You can see how my unfiltered notebook kind of abrubtly ends around connected components. I was able to run everything up until the large star algorithm which threw an exception for a small bug in `col("u").list.map(...)` operation that needed to be adjusted to referencethe wrong column, but since I was still naive to what I was really changing I then transitioned to interactive debugging in a script. Little did I know just how many rabbit holes I was about to go down.

For the next several days I debugged trying the existing approach. At first I was just trying to get the connected components section to run, referencing the provided script as well as the pyspark implementation. I was eventually able to get the workload running, but it yielded no duplicates. Then I was able to yield duplicates, but the number of rows grew in size, instead of reducing. Finally I was seeing the results I was expecting but I was unable to validate them against igraph. The results weren't close. I knew somewhere along the way I had improperly implemented the Large and Small star trasnformations and was having a difficult time identifying where I had gone wrong.

Then, once I got everything to run I was getting MORE rows instead of less.
I iterated and iterated until I saw the numbers of rows get smaller at a scale that made sense. Once I got to this stage, I was pretty excited to think that I was at the finish line, but I wasn't. I knew I wanted to validate my results against a graph processing library and hooked up a validation step to ensure I was getting the right results.

I wasnt... Not even close.

Little did I know how much longer it would take until I would arrive at the correct implementation.

Eventually I ended up implementing Connected Components from first priciples by using the [*Connected Components in MapReduce and Beyond*] paper as a reference.

This was a long and excrutiating process. I honestly can't even begin to explain just how many iterations I tried. Having igraph as a reference was huge. NetworkX is the canonical reference for python, but its probably 20x slower. (iGraph's core is written in C). Without this validation, I would have shipped something that would have "appeared" correct, but wasn't.




#### A genuine moment of artificial intelligence yielding emergent results

Eventually I reached my limit of technical understanding and leaned on multiple independent sessions with gpt-5 and grok 4 to validate my implementation of the algorithms were canonically correct.

Grok-4 was invaluable here. GPT-5 was great and all, but it kept throwing me through this debugging loop adding little operations that I couldn't know weren't a part of the paper. Grok 4 shut that down real fast. With the paper attached it tore my code to pieces and provided me a key insight on handling null values inside the large and small star operations that ended up helping me succeed.

It was the fact that I personally didn't fully understand what "correct" looked like that prevented me from making progress. Each time I would update the pipeline, I just did not know what I was changing.

Using multiple independent sessions outside my coding context also gave me the chance to validate my core understanding of the algorithm. If you work with a model inside Cursor, it will reference the existing code. Since I didn't know if I had corrupted the approach, lauching several chat sessions in ChatGPT and Grok to derive the operations from the paper itself.

As determined as I was to implement the solution myself, it was getting late into the morning and I finally got desperate enough to hand the problem over to an agent to figure out. Since I had igraph to benchmark against, I knew I had a chance of gaining progress, even if was only an insight into a bug in my own code. 

I might not have ever implemented connected components before, but I sure as hell know how to steer an llm agent. I launched gpt-5 and asked it to autonomously iterate approaches until it could demonstrate parity with igraphs results.  The [explanation from GPT-5](/friction/connected_components_reasoning.md) details the solution GPT-5 found in the moment.

You'll find an additional step in my implementation of connected components called `global_min_label_propagation` which has a detailed docstring and comments on how it works. If you look at the operations closely, the `group_by` and neighbor minimization appear extremely similar to the small star operation, but with an important difference.

The `edges.join(labels, left_on="v", right_on="u", how="left")` is not the same thing as a canonicalized or symmetricized edge list. The other difference is that this is repeating this same operation until the label pairs are equal. It does end up taking multiple iterations as I've observed, so its not a one-op silver-bullet.

This global label reduction helps cleanup the differences I was seeing in my igraph parity validation tests. I personally have no idea what the precedent is for an operation like this. Its definitely not in the original paper, but it works and leverages dataframe ops so I'm not sure I care.

Considering the amount of frustration and pain I had in debugging the correctness of the algorithm, I'm just glad I have something that works and scales. The fact that it our daft implmentation is canonically exact to igraph is ... well... amazing!

It's always suspicious when an LLM gives you a "few extra steps" just to get the answer right. A lot of these models are known for spec gaming coding tasks with workarounds or hacks that make them succeed regardless of the outcome, but I have been working with GPT-5 for a while and have developed a strong intuition for it's reliability. I did a thorough review of GPT-5's answers and found it's global label propgation to be genuinely necessary for igaph parity. The other function it added wasn't necessary.

This is probably my first real encounter with artificial intelligence yielding real material emergent results. Naturally I'd love to see some more testing around this to identify if it might become a bottleneck at the trillion document level, but seriously... I'm just happy to be done and have something that works.

### Debugging (saving grace)

I really can't emphasize enough just how much this week's focus was on getting the algorithm right more than daft itself. With a few minor exceptions, daft had pretty much everything I needed to build this solution, it was just a matter of getting everything working and validated.

If anything, daft made it super easy to debug anywhere I needed to. I mentioned this on the slack channel but I really like this wrapped parentheses pattern for aggregating lazy transformations.

I can just drop a breakpoint anywhere in the transformation list, add a `.show()` and inspect the data. I can also quickly add select or where filters to quickly visualize just the data I care about, and the pretty table I get in the terminal is just nice to look at. I know we kind of take it for granted at this point, but when I was working with pandas for converting the data to the igraph constructor, I was surprised at how opaque the prints were.

