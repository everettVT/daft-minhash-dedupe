#struct.get doesn't work? 
import daft
from daft import struct, col

df = daft.from_pydict(
    {"foo": [{"a":1,"b":2}]}
)

df = df.with_column("bar", col("foo").struct.get("a")).collect()

