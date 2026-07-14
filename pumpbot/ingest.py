"""Stream parsing surface (canonical: pumpfun_parse.py). grpc_capture /
grpc_firehose are collector entry points (frozen set): import them directly
only in offline tools, never restart-couple new code to them."""
from ._lazy import make_lazy

make_lazy(__name__, {
    "TradeEvent": ("pumpfun_parse", "TradeEvent"),
    "parse_program_data_line": ("pumpfun_parse", "parse_program_data_line"),
})
