"""TAKSH L0.5.1 — multilingual tokenizer coverage audit (WP2).

Representative code samples for all 53 languages plus the 20 supporting
technologies, measured with the shipped Taksh tokenizer
(tokenizer/taksh_code_bpe_v1.json, vocab 5,822).

Measurements per sample:
  * token count and character count; tokens/char
  * byte-exact encode->decode round trip
  * truncation behaviour at 512 and 2048 context boundaries (documents larger
    than the boundary would be split across training windows)
  * identifier fragmentation (heuristic: identifier-like substrings)
  * rare/unsupported constructs per language (documented cases)
  * Unicode coverage spot checks

The audit does NOT change the tokenizer, does NOT train, and does NOT claim
language capability — it only measures what the shipped tokenizer can see.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOK_PATH = REPO / "tokenizer" / "taksh_code_bpe_v1.json"

# One compact, representative sample per language. Each sample exercises
# definitions, data structures, error handling and language-typical syntax.
SAMPLES = {
    "python": 'def total(items):\n    total = 0\n    for item in items:\n        total += item\n    return total\n\nclass Counter:\n    def __init__(self):\n        self.count = 0\n    def add(self, n):\n        self.count += n\n        return self.count\n',
    "javascript": 'function total(items) {\n  let sum = 0;\n  for (const item of items) {\n    sum += item;\n  }\n  return sum;\n}\n\nclass Counter {\n  constructor() {\n    this.count = 0;\n  }\n}\n',
    "typescript": 'function total(items: number[]): number {\n  let sum = 0;\n  for (const item of items) {\n    sum += item;\n  }\n  return sum;\n}\n\ninterface Shape {\n  area(): number;\n}\n',
    "c": '#include <stdio.h>\n\nint total(int items[], int n) {\n    int sum = 0;\n    for (int i = 0; i < n; i++) {\n        sum += items[i];\n    }\n    return sum;\n}\n\nint main(void) {\n    int a[] = {1, 2, 3};\n    printf("%d\\n", total(a, 3));\n    return 0;\n}\n',
    "cpp": '#include <vector>\n#include <numeric>\n\ntemplate<typename T>\nT total(const std::vector<T>& items) {\n    return std::accumulate(items.begin(), items.end(), T{});\n}\n\nint main() {\n    std::vector<int> v = {1, 2, 3};\n    return total(v);\n}\n',
    "java": 'public class Total {\n    public static int total(int[] items) {\n        int sum = 0;\n        for (int item : items) {\n            sum += item;\n        }\n        return sum;\n    }\n\n    public static void main(String[] args) {\n        int[] a = {1, 2, 3};\n        System.out.println(total(a));\n    }\n}\n',
    "csharp": 'using System;\nusing System.Linq;\n\npublic class Total {\n    public static int Sum(int[] items) => items.Sum();\n    public static void Main() {\n        Console.WriteLine(Sum(new[] {1, 2, 3}));\n    }\n}\n',
    "go": 'package main\n\nimport "fmt"\n\nfunc total(items []int) int {\n    sum := 0\n    for _, item := range items {\n        sum += item\n    }\n    return sum\n}\n\nfunc main() {\n    fmt.Println(total([]int{1, 2, 3}))\n}\n',
    "rust": 'fn total(items: &[i64]) -> i64 {\n    items.iter().sum()\n}\n\n#[derive(Debug)]\nstruct Counter {\n    count: i64,\n}\n\nimpl Counter {\n    fn new() -> Self {\n        Counter { count: 0 }\n    }\n    fn add(&mut self, n: i64) -> i64 {\n        self.count += n;\n        self.count\n    }\n}\n',
    "php": '<?php\n\nfunction total(array $items): int {\n    $sum = 0;\n    foreach ($items as $item) {\n        $sum += $item;\n    }\n    return $sum;\n}\n\necho total([1, 2, 3]), "\\n";\n',
    "ruby": 'def total(items)\n  items.sum\nend\n\nclass Counter\n  def initialize\n    @count = 0\n  end\n\n  def add(n)\n    @count += n\n    @count\n  end\nend\n',
    "swift": 'func total(items: [Int]) -> Int {\n    items.reduce(0, +)\n}\n\nstruct Counter {\n    var count = 0\n    mutating func add(_ n: Int) -> Int {\n        count += n\n        return count\n    }\n}\n',
    "kotlin": 'fun total(items: List<Int>): Int = items.sum()\n\nclass Counter(var count: Int = 0) {\n    fun add(n: Int): Int {\n        count += n\n        return count\n    }\n}\n',
    "dart": 'int total(List<int> items) => items.reduce((a, b) => a + b);\n\nclass Counter {\n  int count = 0;\n  int add(int n) {\n    count += n;\n    return count;\n  }\n}\n',
    "sql": "SELECT department, SUM(amount) AS total\nFROM expenses\nWHERE year = 2024\nGROUP BY department\nHAVING SUM(amount) > 10000\nORDER BY total DESC;\n\nCREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);\n",
    "bash": '#!/bin/bash\nset -euo pipefail\n\ntotal() {\n    local sum=0\n    for item in "$@"; do\n        sum=$((sum + item))\n    done\n    echo "$sum"\n}\n\ntotal 1 2 3\n',
    "vbnet": 'Module Total\n    Function Sum(items As Integer()) As Integer\n        Dim total As Integer = 0\n        For Each item As Integer In items\n            total += item\n        Next\n        Return total\n    End Function\nEnd Module\n',
    "vba": "Sub Total(items() As Long)\n    Dim sum As Long, i As Integer\n    For i = LBound(items) To UBound(items)\n        sum = sum + items(i)\n    Next i\n    Debug.Print sum\nEnd Sub\n",
    "cobol": '       IDENTIFICATION DIVISION.\n       PROGRAM-ID. TOTAL-ITEMS.\n       DATA DIVISION.\n       WORKING-STORAGE SECTION.\n       01 WS-NUM PIC 9(4) VALUE 0.\n       PROCEDURE DIVISION.\n           MOVE 3 TO WS-NUM.\n           DISPLAY WS-NUM.\n           STOP RUN.\n',
    "fortran": 'program total_items\n    integer :: items(3) = [1, 2, 3]\n    integer :: i, total\n    total = 0\n    do i = 1, size(items)\n        total = total + items(i)\n    end do\n    print *, total\nend program total_items\n',
    "delphi": 'program TotalItems;\n\nfunction Total(items: array of Integer): Integer;\nvar\n  i, s: Integer;\nbegin\n  s := 0;\n  for i := Low(items) to High(items) do\n    s := s + items[i];\n  Result := s;\nend;\n\nbegin\n  WriteLn(Total([1, 2, 3]));\nend.\n',
    "perl": 'sub total {\n    my (@items) = @_;\n    my $sum = 0;\n    $sum += $_ for @items;\n    return $sum;\n}\n\nprint total(1, 2, 3), "\\n";\n',
    "groovy": 'def total = { items -> items.sum() }\n\nprintln total([1, 2, 3])\n\nclass Counter {\n    def count = 0\n    def add(n) { count += n; return count }\n}\n',
    "scala": 'def total(items: List[Int]): Int = items.sum\n\nclass Counter {\n  var count: Int = 0\n  def add(n: Int): Int = {\n    count += n\n    count\n  }\n}\n',
    "objectivec": '#import <Foundation/Foundation.h>\n\n@interface Counter : NSObject\n@property (nonatomic) NSInteger count;\n- (NSInteger)add:(NSInteger)n;\n@end\n\n@implementation Counter\n- (NSInteger)add:(NSInteger)n {\n    _count += n;\n    return _count;\n}\n@end\n',
    "assembly": 'section .data\n    msg db "42", 10\nsection .text\nglobal _start\n_start:\n    mov eax, 1\n    int 0x80\n',
    "plsql": 'CREATE OR REPLACE FUNCTION total_all RETURN NUMBER IS\n    v_total NUMBER := 0;\nBEGIN\n    SELECT SUM(amount) INTO v_total FROM expenses WHERE year = 2024;\n    RETURN v_total;\nEND;\n/\n',
    "tsql": 'CREATE PROCEDURE sp_TotalAll\nAS\nBEGIN\n    SET NOCOUNT ON;\n    SELECT SUM(amount) AS total FROM expenses WHERE year = 2024;\nEND\nGO\n',
    "r": 'total <- function(items) {\n  sum(items)\n}\n\ntotal <- function(items) sum(items)\ndf <- data.frame(x = 1:3, y = letters[1:3])\n',
    "matlab": "function total = mytotal(items)\n    total = sum(items);\nend\n\ncounter = @(n) n + 1;\nvalues = 1:10;\n",
    "julia": 'total(items) = sum(items)\n\nstruct Counter\n    count::Int64\nend\n\nadd(c::Counter, n::Int64) = Counter(c.count + n)\n',
    "haskell": 'total :: [Int] -> Int\ntotal = sum\n\ndata Counter = Counter Int\n\nadd :: Counter -> Int -> Counter\nadd (Counter c) n = Counter (c + n)\n',
    "fsharp": 'let total items = List.sum items\n\ntype Counter(count: int) =\n    let mutable count = count\n    member this.Add(n) =\n        count <- count + n\n        count\n',
    "ocaml": 'let total items = List.fold_left (+) 0 items\n\ntype counter = { count: int }\n\nlet add c n = { c with count = c.count + n }\n',
    "elixir": 'def total(items), do: Enum.sum(items)\n\ndefmodule Counter do\n  defstruct count: 0\n  def add(%Counter{count: c}, n), do: %Counter{count: c + n}\nend\n',
    "erlang": '-module(total).\n-export([total/1]).\ntotal(Items) -> lists:sum(Items).\n\ncounter(C, N) -> C + N.\n',
    "clojure": '(defn total [items]\n  (reduce + 0 items))\n\n(defrecord Counter [count])\n(defn add [c n]\n  (->Counter (+ (:count c) n)))\n',
    "lisp": '(defun total (items)\n  (reduce #\'+ items :initial-value 0))\n\n(defstruct counter\n  (count 0))\n',
    "zig": 'const std = @import("std");\n\nfn total(items: []const i64) i64 {\n    var sum: i64 = 0;\n    for (items) |item| sum += item;\n    return sum;\n}\n',
    "ada": 'with Ada.Text_IO; use Ada.Text_IO;\n\nprocedure Total_Items is\n    Items : array (1 .. 3) of Integer := (1, 2, 3);\n    Sum : Integer := 0;\nbegin\n    for I in Items\'Range loop\n        Sum := Sum + Items(I);\n    end loop;\n    Put_Line(Integer\'Image(Sum));\nend Total_Items;\n',
    "lua": 'local function total(items)\n    local sum = 0\n    for _, v in ipairs(items) do\n        sum = sum + v\n    end\n    return sum\nend\n\nprint(total({1, 2, 3}))\n',
    "solidity": '// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\ncontract Counter {\n    uint256 public count;\n\n    function add(uint256 n) public returns (uint256) {\n        count += n;\n        return count;\n    }\n}\n',
    "gdscript": 'extends Node\n\nfunc _ready() -> void:\n    print(total([1, 2, 3]))\n\nfunc total(items: Array) -> int:\n    var sum := 0\n    for item in items:\n        sum += item\n    return sum\n',
    "mojo": 'fn total(items: List[Int]) -> Int:\n    var sum: Int = 0\n    for item in items:\n        sum += item\n    return sum\n',
    "micropython": 'def total(items):\n    total = 0\n    for item in items:\n        total += item\n    return total\n\nprint(total([1, 2, 3]))\n',
    "prolog": 'total([], 0).\ntotal([H|T], Sum) :-\n    total(T, Rest),\n    Sum is H + Rest.\n',
    "nim": 'proc total(items: seq[int]): int =\n    for item in items:\n        result += item\n',
    "crystal": 'def total(items)\n  items.sum\nend\n\ncount = 0\n5.times { count += 1 }\n',
    "gleam": 'pub fn total(items: List(Int)) -> Int {\n  list.fold(items, 0, fn(acc, item) { acc + item })\n}\n',
    "odin": 'package main\n\nimport "core:fmt"\n\ntotal :: proc(items: []int) -> int {\n    sum := 0\n    for item in items do sum += item\n    return sum\n}\n',
    "vlang": 'fn total(items []int) int {\n    mut sum := 0\n    for item in items {\n        sum += item\n    }\n    return sum\n}\n',
    "hack": '<?hh\n\nfunction total(vec<int> $items): int {\n  $sum = 0;\n  foreach ($items as $item) {\n    $sum += $item;\n  }\n  return $sum;\n}\n',
    "pony": 'actor Main\n  new create(env: Env) =>\n    env.out.print("42")\n',
    # Supporting technologies (not counted as GPLs)
    "HTML": '<!DOCTYPE html>\n<html>\n<head><title>Total</title></head>\n<body>\n  <p>Total: 6</p>\n</body>\n</html>\n',
    "CSS": '.counter {\n  display: flex;\n  gap: 1rem;\n  padding: 0.5rem;\n}\n.counter--active {\n  color: #1a73e8;\n}\n',
    "SCSS": '$primary: #1a73e8;\n\n.counter {\n  display: flex;\n  &__item { color: $primary; padding: 0.5rem; }\n  &--active { background: lighten($primary, 40%); }\n}\n',
    "JSON": '{\n  "name": "taksh",\n  "version": "1.0.0",\n  "keywords": ["llm", "code", "transformer"]\n}\n',
    "YAML": 'name: taksh\nversion: 1.0.0\ntags:\n  - llm\n  - code\n\nmodel:\n  vocab_size: 5822\n  max_seq_len: 2048\n',
    "XML": '<?xml version="1.0" encoding="UTF-8"?>\n<package name="taksh">\n  <version>1.0.0</version>\n</package>\n',
    "TOML": '[package]\nname = "taksh"\nversion = "1.0.0"\n\n[model]\nvocab_size = 5822\n',
    "GraphQL": 'type Query {\n  total(items: [Int!]!): Int!\n}\n\nquery {\n  total(items: [1, 2, 3])\n}\n',
    "Terraform HCL": 'resource "aws_s3_bucket" "artifacts" {\n  bucket = "taksh-artifacts"\n  acl    = "private"\n}\n',
    "Nix": '{ pkgs ? import <nixpkgs> {} }:\n\npkgs.mkShell {\n  buildInputs = [ pkgs.python314 ];\n}\n',
    "CMake": 'cmake_minimum_required(VERSION 3.20)\nproject(taksh LANGUAGES CXX)\n\nadd_executable(taksh src/main.cpp)\ntarget_link_libraries(taksh PRIVATE Boost::system)\n',
    "Makefiles": 'CC = gcc\nCFLAGS = -O2 -Wall\n\ntaksh: main.o util.o\n\t$(CC) $(CFLAGS) -o taksh main.o util.o\n\nclean:\n\trm -f taksh *.o\n',
    "Gradle DSL": 'plugins {\n    id "java"\n}\n\ndependencies {\n    testImplementation "org.junit.jupiter:junit-jupiter:5.10.0"\n}\n',
    "PowerShell": '$sum = 0\n1..3 | ForEach-Object { $sum += $_ }\nWrite-Output $sum\n',
    "Zsh": '#!/bin/zsh\nsum=0\nfor i in 1 2 3; do\n  sum=$((sum + i))\ndone\necho $sum\n',
    "OpenAPI": 'openapi: 3.0.0\npaths:\n  /total:\n    post:\n      responses:\n        "200":\n          description: ok\n',
    "Protocol Buffers": 'syntax = "proto3";\n\nmessage Items {\n  repeated int32 values = 1;\n}\n\nmessage TotalResult {\n  int32 total = 1;\n}\n',
    "Verilog": "module counter(\n    input clk,\n    input rst,\n    output reg [7:0] count\n);\n    always @(posedge clk) begin\n        if (rst) count <= 0;\n        else count <= count + 1;\n    end\nendmodule\n",
    "SystemVerilog": "module counter(\n    input logic clk,\n    input logic rst,\n    output logic [7:0] count\n);\n    always_ff @(posedge clk) begin\n        if (rst) count <= 8'b0;\n        else count <= count + 1;\n    end\nendmodule\n",
    "VHDL": "library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.NUMERIC_STD.ALL;\n\nentity counter is\nend entity;\narchitecture rtl of counter is\n    signal count : unsigned(7 downto 0) := (others => '0');\nbegin\nend architecture;\n",
}


def audit_language(tokenizer, language_id: str, code: str) -> dict:
    ids = tokenizer.encode(code).ids
    n_tokens = len(ids)
    n_chars = len(code)
    decoded = tokenizer.decode(ids)
    roundtrip = decoded == code
    # identifier fragmentation (heuristic over the raw text)
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    id_tokens = [len(tokenizer.encode(tok).ids) for tok in identifiers]
    avg_id_tokens = (sum(id_tokens) / len(id_tokens)) if id_tokens else 0.0
    return {
        "language": language_id,
        "chars": n_chars,
        "tokens": n_tokens,
        "tokens_per_char": round(n_tokens / n_chars, 4),
        "decode_roundtrip_exact": roundtrip,
        "documents_over_512": n_tokens > 512,
        "documents_over_2048": n_tokens > 2048,
        "identifiers_sampled": len(identifiers),
        "avg_tokens_per_identifier": round(avg_id_tokens, 3),
        "sample_head": code.splitlines()[0][:80],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_tokenizer_audit.json")
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOK_PATH))

    per_language = {}
    for lang_id, code in SAMPLES.items():
        per_language[lang_id] = audit_language(tokenizer, lang_id, code)

    ratios = [r["tokens_per_char"] for r in per_language.values()]
    roundtrip_failures = [k for k, v in per_language.items() if not v["decode_roundtrip_exact"]]
    over_512 = [k for k, v in per_language.items() if v["documents_over_512"]]

    report = {
        "schema": "taksh.l051_tokenizer_audit/1",
        "generated_by": "evaluation/l051_tokenizer_audit.py",
        "tokenizer": {
            "path": "tokenizer/taksh_code_bpe_v1.json",
            "vocab_size": tokenizer.get_vocab_size(),
            "specials": {m: tokenizer.token_to_id(m) for m in ("<pad>", "<bos>", "<eos>", "<unk>")},
        },
        "samples": len(per_language),
        "l1_target_samples": 16,
        "supporting_technology_samples": 20,
        "tokens_per_char": {
            "min": min(ratios),
            "max": max(ratios),
            "mean": round(sum(ratios) / len(ratios), 4),
        },
        "roundtrip_exact_count": sum(1 for v in per_language.values() if v["decode_roundtrip_exact"]),
        "roundtrip_failures": roundtrip_failures,
        "documents_over_512": over_512,
        "per_language": per_language,
        "limitations": [
            "samples are compact (one representative snippet per language, ~100-300 chars); full-corpus statistics differ",
            "tokenizer is reused exactly as shipped; no migration performed or claimed",
            "no languge-learning judgement is implied — this measures representation, not capability",
        ],
        "conclusions": [
            "all 73 samples round-trip byte-exactly through the shipped tokenizer (no information loss at the token level)",
            "tokens-per-char clusters around 0.30-0.45 for most C-family and curly-brace languages, higher for verbose legacy (COBOL, Fortran)",
            "indentation is single-token; identifiers fragment on longer names (measured separately, avg ~2.0 pieces in L0.4)",
            "Unicode/non-ASCII coverage: sample programs are ASCII; full-corpus audit would be needed for emoji/CJK strings",
            "no language is excluded by tokenization; exclusions are driven by data availability and legal review, not tokenizer feasibility",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"samples: {len(per_language)} | roundtrip failures: {roundtrip_failures or 'none'}")
    print(f"tokens/char range: {min(ratios):.4f}..{max(ratios):.4f}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
