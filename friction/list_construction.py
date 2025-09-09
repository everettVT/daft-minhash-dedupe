import daft 
from daft import col 
from selectolax.parser import HTMLParser

df = daft.from_pydict({
    "u": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "v": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
})


# CASE 1: Large/Small Star Reduction following canonicalization/symmetrization

# Reduce ⟨u; N ⊆ Γ(u)⟩:
df = df.with_column("neigh", df.groupby("u").agg_list("v"))

# m = argminv∈Γ+(u) lv - with list append (intuitive)
df_list_append = df.with_column("m", col("neigh").list.append(col("u")).list.min())


# m = argminv∈Γ+(u) lv - multi-step process
df_multi_step = df.with_column("m", col("nbrs").list.min())
df_multi_step = df_multi_step.with_column(
    "m", 
    col("m").is_null().if_else(
        col("u"),
        (col("u") < col("m")).if_else(col("u"), col("m"))
    )
)

# Emit stage stays the same (we can't avoid explosion)


# CASE 2: Building a deterministed block index following html text extraction

#  Context: For each html record, we extract a list of text blocks from desired tags. 
uri = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-33/segments/1754151279521.11/warc/CC-MAIN-20250802220907-20250803010907-00000.warc.gz"
row_limit = 1000
index_col = "block_id"
content_col = "block_text"
df_warc = daft.read_warc(uri).limit(row_limit)

@daft.func()
def remove_http_headers(x: str) -> str:
    """Remove HTTP headers from input string by splitting on double CRLF, returning the body or empty string."""
    if x is None:
        return ""
    if len(x.split("\r\n\r\n")) > 1:
        return x.split("\r\n\r\n")[1]
    return ""

@daft.func()
def extract_blocks(html: str) -> list[str]:
    """Parse HTML using Selectolax, remove scripts/styles/noscripts, extract text blocks from key tags."""
    tree = HTMLParser(html)
    for n in tree.css("script,style,noscript"):
        n.decompose()

    blocks = []
    for node in tree.css("""title, article, main, p, h1, h2, h3, h4, h5, h6, li, div, section, img[alt], figcaption, caption, blockquote, table th, table td, pre, code, summary, meta[name="description"], meta[property="og:title"], meta[property="og:description"]"""):
        txt = node.text(separator=" ", strip=True)
        if txt: 
            blocks.append(txt)
    return blocks

# \\ Current implementation // <------ UDF of Focus
@daft.func()
def get_block_idx(blocks: list[str]) -> list[int]:
    """Generate integer indices for each element in the input list of blocks."""
    return list(range(len(blocks)))

df_html = (
    df_warc
    .where(col("WARC-Identified-Payload-Type")== "text/html")
    .with_column("content_raw", remove_http_headers(col("warc_content").try_decode("utf-8")))
    .where(col("content_raw") != "")
)  

df_text = (
    df_html
    .with_column("blocks", extract_blocks(col("content_raw")))
    .with_column("block_idx", get_block_idx(col("blocks"))) # <----- Implemented Here
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


