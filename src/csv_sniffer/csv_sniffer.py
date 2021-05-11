"""
Sniffer for Pandas csv_read, allows interactive specification of data types,
names, and various parameters before loading the whole file.
"""
import csv
import inspect
import io

import pandas as pd
import numpy as np
import fsspec
from ipywidgets import widgets

# from traitlets import HasTraits, observe, Instance


def quote_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_parser_defaults = {
    key: val.default
    for key, val in inspect.signature(pd.read_csv).parameters.items()
    if val.default is not inspect._empty
}

# Borrowed from pandas
MANDATORY_DIALECT_ATTRS = (
    "delimiter",
    "doublequote",
    "escapechar",
    "skipinitialspace",
    "quotechar",
    "quoting",
)


def _merge_with_dialect_properties(dialect, defaults):
    if not dialect:
        return defaults
    kwds = defaults.copy()

    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)

        parser_default = _parser_defaults[param]
        provided = kwds.get(param, parser_default)

        # Messages for conflicting values between the dialect
        # instance and the actual parameters provided.
        conflict_msgs = []

        # Don't warn if the default parameter was passed in,
        # even if it conflicts with the dialect (gh-23761).
        if provided != parser_default and provided != dialect_val:
            msg = (
                f"Conflicting values for '{param}': '{provided}' was "
                f"provided, but the dialect specifies '{dialect_val}'. "
                "Using the dialect-specified value."
            )

            # Annoying corner case for not warning about
            # conflicts between dialect and delimiter parameter.
            # Refer to the outer "_read_" function for more info.
            if not (param == "delimiter" and kwds.pop("sep_override", False)):
                conflict_msgs.append(msg)

        if conflict_msgs:
            print("\n\n".join(conflict_msgs))
        kwds[param] = dialect_val
    return kwds


def _force_list(series):
    if isinstance(series, list):
        return series
    return [series]


class CSVSniffer:
    """
    Jupyter notebook component to assist in specifying parameters
    to the pandas.load_csv function.
    """

    signature = inspect.signature(pd.read_csv)
    delimiters = [",", ";", "<TAB>", "<SPACE>", ":", "skip initial space"]
    del_values = [",", ";", "\t", " ", ":", "skip"]

    def __init__(self, path, lines=100, cache_storage=None, **args):
        self.path = path
        self._args = args
        self._cache_storage = cache_storage
        self.lines = widgets.BoundedIntText(
            value=lines,
            min=10,
            max=1000,
            continuous_update=False,
            description="Lines:",
        )
        self.lines.observe(self._lines_cb, names="value")
        self._head = ""
        self._dialect = None
        self.params = {}
        self._df = None
        self._df2 = None
        self._dtypes = None
        self._usecols = None
        self._names = None
        # Widgets
        layout = widgets.Layout(border="solid")
        self.head_text = widgets.HTML()
        self.df_text = widgets.HTML()
        self.df2_text = widgets.HTML()
        self.error_msg = widgets.Textarea(description="Error:")
        self.tab = widgets.Tab(
            [self.head_text, self.df_text, self.df2_text],
            layout=widgets.Layout(max_height="1024px"),
        )
        for i, title in enumerate(["Head", "DataFrame", "DataFrame2"]):
            self.tab.set_title(i, title)
        # Delimiters
        self.delimiter = widgets.RadioButtons(
            orientation="horizontal",
            options=list(zip(self.delimiters, self.del_values)),
        )
        self.delimiter.observe(self._delimiter_cb, names="value")
        self.delim_other = widgets.Text()
        self.delim_other.observe(self._delimiter_cb, names="value")
        self.delimiter = widgets.VBox(
            [
                self.delimiter,
                self.delim_other,
            ],
            layout=layout,
        )
        # Dates
        # TODO
        self.dayfirst = widgets.Checkbox(description="Dayfirst", value=False)
        self.date_parser = widgets.Text(description="Date parser:", value="")
        self.infer_datetime = widgets.Checkbox(
            description="Infer datetime", value=False
        )
        self.date = widgets.VBox(
            [self.dayfirst, self.infer_datetime, self.date_parser],
            layout=layout,
        )
        # Header
        self.header = widgets.BoundedIntText(
            value=-1,
            min=-1,
            max=1000,
            continuous_update=False,
            description="Header:",
        )
        self.skiprows = widgets.BoundedIntText(
            value=0,
            min=0,
            max=1000,
            continuous_update=False,
            description="Skip rows:",
        )
        self.skiprows.observe(self._skiprows_cb, names="value")
        # Special values
        self.true_values = widgets.Text(
            description="True values", continuous_update=False
        )
        self.true_values.observe(self._true_values_cb, names="value")
        self.false_values = widgets.Text(
            description="False values", continuous_update=False
        )
        self.false_values.observe(self._false_values_cb, names="value")
        self.na_values = widgets.Text(
            description="NA values", continuous_update=False
        )
        self.na_values.observe(self._na_values_cb, names="value")
        self.special_values = widgets.VBox(
            [self.true_values, self.false_values, self.na_values],
            layout=layout,
        )

        # Global tab with Delimiters and Dates
        self.global_tab = widgets.Tab(
            [
                self.delimiter,
                self.date,
                widgets.VBox(
                    [self.lines, self.header, self.skiprows], layout=layout
                ),
                self.special_values,
            ]
        )
        for i, title in enumerate(
            ["Delimiters", "Dates", "Header", "Special values"]
        ):
            self.global_tab.set_title(i, title)

        # Column selection
        self.columns = widgets.SelectMultiple(disabled=True, rows=15)
        self.columns.observe(self._columns_cb, names="value")
        # Column details
        self.no_detail = widgets.Label(value="No Column Selected")
        self.details = widgets.Box([self.no_detail], label="Details")
        # Toplevel Box
        self.top = widgets.HBox(
            [
                self.global_tab,
                widgets.VBox(
                    [
                        widgets.Label("Columns"),
                        self.columns,
                    ],
                    layout=layout,
                ),
                widgets.VBox(
                    [widgets.Label("Selected Column"), self.details],
                    layout=layout,
                ),
            ]
        )
        self.cmdline = widgets.Textarea(
            layout=widgets.Layout(width="100%"), rows=3
        )
        self.testBtn = widgets.Button(description="Test")
        self.box = widgets.VBox(
            [
                self.top,
                widgets.HBox(
                    [
                        self.testBtn,
                        widgets.Label(value="CmdLine:"),
                        self.cmdline,
                    ]
                ),
                self.tab,
            ]
        )
        self.testBtn.on_click(self.test_cmd)
        self.column_info = None
        self.clear()
        self.dataframe()

    def _parse_list(self, key, values):
        split = [s for s in values.split(",") if s]
        if split:
            self.params[key] = split
        else:
            self.params.pop(key, None)
        self.set_cmdline()

    def _true_values_cb(self, change):
        self._parse_list("true_values", change["new"])

    def _false_values_cb(self, change):
        self._parse_list("false_values", change["new"])

    def _na_values_cb(self, change):
        self._parse_list("na_values", change["new"])

    def _skiprows_cb(self, change):
        skip = change["new"]
        self._head = ""
        if skip == 0:
            self.params.pop("skiprows", False)
        else:
            self.params["skiprows"] = skip
        self.set_cmdline()
        self.dataframe(force=True)

    def _lines_cb(self, change):
        self._head = ""
        self.dataframe(force=True)

    def _delimiter_cb(self, change):
        delim = change["new"]
        # print(f"Delimiter: '{delim}'")
        self.set_delimiter(delim)

    def _columns_cb(self, change):
        column = change["new"]
        if not column:
            return
        if len(column) == 1:
            column = column[0]
            self.show_column(column)
        else:
            self.show_columns(column)

    def set_delimiter(self, delim):
        if delim == "skip":
            delim = " "
            if self.params.get("skipinitialspace"):
                return
            self.params["skipinitialspace"] = True
        self._dialect.delimiter = delim  # TODO check valid delim
        self.delim_other.value = delim
        self.delimiter.value = delim
        self.tab.selected_index = 1
        if self._df is not None:
            self._reset()
        else:
            self.params = _merge_with_dialect_properties(
                self._dialect, self.params
            )
        self.dataframe(force=True)

    def _reset(self):
        args = self._args.copy()
        self.params = {}
        # TODO can we keep them somehow?
        self._usecols = None
        self._names = None
        for name, param in self.signature.parameters.items():
            if name != "sep" and param.default is not inspect._empty:
                self.params[name] = args.pop(name, param.default)
        self.params["index_col"] = False
        self.params = _merge_with_dialect_properties(
            self._dialect, self.params
        )
        self.set_cmdline()
        if args:
            raise ValueError(f"extra keywords arguments {args}")

    def kwargs(self):
        "Return the arguments to pass to pandas.csv_read"
        # First, purge the columns not used from dtype and parse_dates
        params = {}
        for key, val in self.params.items():
            default = _parser_defaults[key]
            if val == default:
                continue
            params[key] = val
        return params

    def set_cmdline(self):
        params = self.kwargs()
        self.cmdline.value = str(params)

    def clear(self):
        self.lines.value = 100
        self._head = ""
        self.head_text.value = '<pre style="white-space: pre"></pre>'
        self.df_text.value = ""
        self._dialect = None
        self._reset()

    def _format_head(self):
        self.head_text.value = (
            '<pre style="white-space: pre">'
            + quote_html(self._head)
            + "</pre>"
        )

    def head(self):
        if self._head:
            return self._head
        if self._cache_storage:
            filecache = {"filecache": {"cache_storage": self.cache_storage}}
        else:
            filecache = {}
        with fsspec.open(
            self.path, mode="rt", compression="infer", **filecache
        ) as inp:
            lineno = 0
            # TODO assumes that newline is correctly specified to fsspec
            for line in inp:
                if line and lineno < self.lines.value:
                    self._head += line
                    lineno += 1
                else:
                    break
        self._format_head()
        return self._head

    def dialect(self, force=False):
        if not force and self._dialect:
            return self._dialect
        sniffer = csv.Sniffer()
        head = self.head()
        self._dialect = sniffer.sniff(head)
        # self.params['dialect'] = self._dialect
        self.set_delimiter(self._dialect.delimiter)
        if self.params["header"] == "infer":
            if sniffer.has_header(head):
                self.params["header"] = 0
                self.header.value = 0
            else:
                self.header.value = -1
                self.params["header"] = None
        return self._dialect

    def dataframe(self, force=False):
        if not force and self._df is not None:
            return self._df
        self.dialect()
        strin = io.StringIO(self.head())
        try:
            # print(f"read_csv params: {self.params}")
            self._df = pd.read_csv(strin, **self.params)
        except ValueError as e:
            self._df = None
            self.df_text.value = f"""
<pre style="white-space: pre">Error {quote_html(repr(e))}</pre>
"""
        else:
            with pd.option_context(
                "display.max_rows", self.lines.value, "display.max_columns", 0
            ):
                self.df_text.value = self._df._repr_html_()
        self.dataframe_to_columns()
        self.dataframe_to_params()
        return self._df

    def test_cmd(self, button):
        strin = io.StringIO(self.head())
        try:
            self._df2 = pd.read_csv(strin, **self.params)
        except ValueError as e:
            self._df2 = None
            self.df2_text.value = f"""
<pre style="white-space: pre">Error {quote_html(repr(e))}</pre>
"""
        else:
            with pd.option_context(
                "display.max_rows", self.lines.value, "display.max_columns", 0
            ):
                self.df2_text.value = self._df2._repr_html_()
        self.tab.selected_index = 2

    def dataframe_to_params(self):
        df = self._df
        if df is None:
            return
        # if self.params['names'] is None:
        #     self.params['names'] = list(df.columns)
        # # TODO test for existence?
        self.set_cmdline()

    def dataframe_to_columns(self):
        df = self._df
        if df is None:
            self.columns.options = []
            self.columns.disabled = True
            return
        # Lazy creation of columns
        self.columns.options = [str(c) for c in df.columns]
        self.columns.disabled = False
        self.columns.value = [str(df.columns[0])]

    def get_column(self, column):
        df = self._df
        if column in df.columns:
            return ColumnInfo(self, df[column])
        try:
            column = int(column)
            return ColumnInfo(self, df[column])
        except Exception:
            pass
        return None

    def show_column(self, column):
        col = self.get_column(column)
        self.column_info = col
        if col is None:
            # print(f"Not in columns: '{column}'")
            self.details.children = [self.no_detail]
            return
        self.details.children = [col.box]

    def show_columns(self, columns):
        if len(columns) == 1:
            return self.show_column(columns[0])
        try:
            col = ColumnsInfo(self, [self._df[column] for column in columns])
        except Exception:
            col = ColumnsInfo(
                self, [self._df[int(column)] for column in columns]
            )
        self.column_info = col
        self.details.children = [col.box]

    def rename_columns(self, col):
        series = col.series
        if isinstance(series, list):
            series = series[0]
        columns = list(self._df.columns)
        if self._names is None:
            self._names = columns.copy()
        index = columns.index(series.name)
        self._names[index] = col.rename.value
        if self._names != columns:
            self.params["names"] = self._names
        else:
            self.params.pop("names", False)
        self.set_cmdline()

    def get_column_name(self, col):
        names = self.params.get("names", None)
        if names is None:
            return col.name
        if isinstance(col.name, np.int):
            return names[col.name]
        idx = self._df.columns.index(col.name)
        return names[idx]

    def usecols_columns(self, col):
        series = _force_list(col.series)
        cols = {s.name for s in series}
        if self._usecols is None:
            usecols = set(self._df.columns)
        else:
            usecols = self._usecols
        if col.use.value:
            usecols.update(cols)
            self.retype_columns(col)  # restore changed dtype
        else:
            usecols.difference_update(cols)
            self.remove_dtype(col)
        if usecols == set(self._df.columns):
            self.params.pop("usecols", False)
        else:
            self.params["usecols"] = list(usecols)
        self.set_cmdline()

    def is_column_used(self, col):
        usecols = self.params.get("usecols", None)
        if not usecols:
            return True
        return col.name in usecols

    def retype_columns(self, col):
        series_list = _force_list(col.series)
        types = self.params.get("dtype") or {}
        parse_dates = self.params.get("parse_dates") or []
        for series in series_list:
            name = series.name
            typename = col.retype.value or series.dtype.name
            if name in parse_dates:
                parse_dates.remove(name)
            if series.dtype.name != typename:
                if typename == "datetime":
                    types[name] = "str"
                    parse_dates.append(name)
                else:
                    types[name] = typename
            else:
                types.pop(name, False)
        if types:
            self._dtypes = types
            self.params["dtype"] = types
        else:
            self._dtypes = None
            self.params.pop("dtype", False)
        if parse_dates:
            self.params["parse_dates"] = parse_dates
        else:
            self.params.pop("parse_dates", False)
        self.set_cmdline()

    def remove_dtype(self, col):
        series_list = _force_list(col.series)
        types = self.params.get("dtype") or {}
        parse_dates = self.params.get("parse_dates") or []
        for series in series_list:
            name = series.name
            types.pop(name, False)
            if name in parse_dates:
                parse_dates.remove(name)
        if types:
            self._dtypes = types
            self.params["dtype"] = types
        else:
            self._dtypes = None
            self.params.pop("dtype", False)
        if parse_dates:
            self.params["parse_dates"] = parse_dates
        else:
            self.params.pop("parse_dates", False)
        self.set_cmdline()

    def get_column_dtype(self, series):
        if self._dtypes and series.name in self._dtypes:
            dtype = self._dtypes[series.name]
            if dtype == "str" and self.is_column_date(series):
                return "datetime"
            return dtype
        return ""

    def is_column_date(self, col):
        parse_dates = self.params.get("parse_dates", None)
        return parse_dates and col.name in parse_dates

    def load_dataframe(self):
        "Full load the DataFrame with the GUI parameters"
        return pd.read_csv(self.path, **self.params)


class ColumnInfo:
    numeric_types = [
        "",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
        "str",
        "object",
    ]
    object_types = ["", "object", "str", "category", "datetime"]

    def __init__(self, sniffer, series):
        self.sniffer = sniffer
        self.series = series
        self.name = widgets.Text(
            description="Name:", continuous_update=False, disabled=True
        )
        self.rename = widgets.Text(
            description="Rename:", continuous_update=False
        )
        self.type = widgets.Text(
            description="Type:", continuous_update=False, disabled=True
        )
        self.retype = widgets.Dropdown(description="Retype:")
        self.use = widgets.Checkbox(description="Use", value=True)
        label_layout = widgets.Layout(border="solid 1px", width="100%")
        self.nunique = widgets.Label(layout=label_layout)
        self.min = widgets.Label(layout=label_layout)
        self.max = widgets.Label(layout=label_layout)
        self.box = widgets.VBox()
        self.box.children = [
            self.name,
            self.rename,
            self.type,
            self.retype,
            self.use,
            widgets.HBox([widgets.Label(value="Unique vals:"), self.nunique]),
            widgets.HBox(
                [
                    widgets.Label(value="Min:"),
                    self.min,
                    widgets.Label(value="Max:"),
                    self.max,
                ]
            ),
        ]
        self._init_values()
        self.use.observe(self.usecols_column, names="value")
        self.rename.observe(self.rename_column, names="value")
        self.retype.observe(self.retype_column, names="value")

    def _init_values(self):
        sniffer = self.sniffer
        series = self.series
        self.name.value = str(series.name)
        self.rename.value = str(sniffer.get_column_name(series))
        dtype = series.dtype
        self.type.value = dtype.name
        self.retype.options = self.retype_values(dtype.name)
        self.retype.value = sniffer.get_column_dtype(series)
        self.use.value = sniffer.is_column_used(series)
        self.nunique.value = f"{series.nunique()}/{len(series)}"
        self.min.value = str(series.min())
        self.max.value = str(series.max())

    def retype_values(self, typename):
        if typename in self.object_types:
            return self.object_types
        if typename in self.numeric_types:
            return self.numeric_types
        return typename

    def rename_column(self, change):
        self.sniffer.rename_columns(self)

    def usecols_column(self, change):
        self.sniffer.usecols_columns(self)

    def _test_column_type(self, newtype):
        try:
            self.series.as_type(newtype)
        except ValueError as e:
            return e
        return None

    def retype_column(self, change):
        self.sniffer.retype_columns(self)


class ColumnsInfo(ColumnInfo):
    def _init_values(self):
        series = self.series
        assert isinstance(series, list)
        self.name.value = ", ".join([str(s.name) for s in series])
        self.rename.value = self.name.value
        dtypes = set()
        for s in series:
            dtypes.add(s.dtype)
        if len(dtypes) == 1:
            dtype = dtypes.pop()
            self.type.value = dtype.name
            self.retype.options = self.retype_values(dtype.name)
            self.min.value = str(min([s.min() for s in series]))
            self.max.value = str(max([s.max() for s in series]))
        else:
            dtype = np.result_type(*dtypes)
            self.type.value = str(dtype.name)
            try:
                mins = [np.cast[dtype](s.min()) for s in series]
                self.min.value = str(min(mins))
            except Exception:
                self.min.value = "?"
            try:
                maxs = [np.cast[dtype](s.max()) for s in series]
                self.min.value = str(max(maxs))
            except Exception:
                self.max.value = "?"
        # retype
        self.retype.options = self.retype_values(dtype.name)
        retypes = set()
        for s in series:
            retypes.add(self.sniffer.get_column_dtype(s))
        if len(retypes) == 1:
            self.retype.value = list(retypes)[0]
            self.retype.disabled = False
        else:
            self.retype.value = ""
            self.retype.disabled = True
        self.nunique.value = ""
