#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

See the [README.md] file for more information.
"""

import argparse
import functools
import gzip
import hashlib
import io
import math
import os
import portalocker
import re
import sys
import unicodedata
import urllib.request

from collections import Counter, namedtuple
from itertools import zip_longest
from typing import List, Iterable, Tuple, Union

VERSION = "1.4.2"

try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL

    signal(SIGPIPE, SIG_DFL)

except ImportError:
    print("Could not import signal.SIGPIPE (this is expected on Windows machines)")

# Where to store downloaded test sets.
# Define the environment variable $SACREBLEU, or use the default of ~/.sacrebleu.
#
# Querying for a HOME environment variable can result in None (e.g., on Windows)
# in which case the os.path.join() throws a TypeError. Using expanduser() is
# a safe way to get the user's home folder.
USERHOME = os.path.expanduser("~")
SACREBLEU_DIR = os.environ.get("SACREBLEU", os.path.join(USERHOME, ".sacrebleu"))

# n-gram order. Don't change this.
NGRAM_ORDER = 4

# Default values for CHRF
CHRF_ORDER = 6
# default to 2 (per http://www.aclweb.org/anthology/W16-2341)
CHRF_BETA = 2

# The default floor value to use with `--smooth floor`
SMOOTH_VALUE_DEFAULT = 0.0

# This defines data locations.
# At the top level are test sets.
# Beneath each test set, we define the location to download the test data.
# The other keys are each language pair contained in the tarball, and the respective locations of the source and reference data within each.
# Many of these are *.sgm files, which are processed to produced plain text that can be used by this script.
# The canonical location of unpacked, processed data is $SACREBLEU_DIR/$TEST/$SOURCE-$TARGET.{$SOURCE,$TARGET}

# Detailed document metadata annotation in form DocumentID -> CountryCode - Domain - OptionalFinegrainedCountryCode
# While the annotation is subjective with many unclear cases, it may provide useful insights
# when applied on large data (TODO: annotate all documents from recent WMT years, at least for origlang=en, consider renaming "world" to "other").
SUBSETS = {
    "wmt18": "rt.com.68098=US-crime guardian.181611=US-politics bbc.310963=GB-sport washpost.116881=US-politics scotsman.104228=GB-sport timemagazine.75207=OTHER-world-ID "
    "euronews-en.117981=OTHER-crime-AE smh.com.au.242810=US-crime msnbc.53726=US-politics euronews-en.117983=US-politics msnbc.53894=US-crime theglobeandmail.com.62700=US-business "
    "bbc.310870=OTHER-world-AF reuters.196698=US-politics latimes.231739=US-sport thelocal.51929=OTHER-world-SE cbsnews.198694=US-politics reuters.196718=OTHER-sport-RU "
    "abcnews.255599=EU-sport nytimes.127256=US-entertainment scotsman.104225=GB-politics dailymail.co.uk.233026=GB-scitech independent.181088=GB-entertainment "
    "brisbanetimes.com.au.181614=OTHER-business-AU washpost.116837=US-politics dailymail.co.uk.232928=GB-world thelocal.51916=OTHER-politics-IT bbc.310871=US-crime "
    "nytimes.127392=EU-business-DE euronews-en.118001=EU-scitech-FR washpost.116866=OTHER-crime-MX dailymail.co.uk.233025=OTHER-scitech-CA latimes.231829=US-crime "
    "guardian.181662=US-entertainment msnbc.53731=US-crime rt.com.68127=OTHER-sport-RU latimes.231782=US-business latimes.231840=US-sport reuters.196711=OTHER-scitech "
    "guardian.181666=GB-entertainment novinite.com.24019=US-politics smh.com.au.242750=OTHER-scitech guardian.181610=US-politics telegraph.364393=OTHER-crime-ZA "
    "novinite.com.23995=EU-world dailymail.co.uk.233028=GB-scitech independent.181071=GB-sport telegraph.364538=GB-scitech timemagazine.75193=US-politics "
    "independent.181096=US-entertainment upi.140602=OTHER-world-AF bbc.310946=GB-business independent.181052=EU-sport ",
    "wmt19": "bbc.381790=GB-politics rt.com.91337=OTHER-politics-MK nytimes.184853=US-world upi.176266=US-crime guardian.221754=GB-business dailymail.co.uk.298595=GB-business "
    "cnbc.com.6790=US-politics nytimes.184837=OTHER-world-ID upi.176249=GB-sport euronews-en.153835=OTHER-world-ID dailymail.co.uk.298732=GB-crime telegraph.405401=GB-politics "
    "newsweek.51331=OTHER-crime-CN abcnews.306815=US-world cbsnews.248384=US-politics reuters.218882=GB-politics cbsnews.248387=US-crime abcnews.306764=OTHER-world-MX "
    "reuters.218888=EU-politics bbc.381780=GB-crime bbc.381746=GB-sport euronews-en.153800=EU-politics bbc.381679=GB-crime bbc.381735=GB-crime newsweek.51338=US-world "
    "bbc.381765=GB-crime cnn.304489=US-politics reuters.218863=OTHER-world-ID nytimes.184860=OTHER-world-ID cnn.304404=US-crime bbc.381647=US-entertainment "
    "abcnews.306758=OTHER-politics-MX cnbc.com.6772=US-business reuters.218932=OTHER-politics-MK upi.176251=GB-sport reuters.218921=US-sport cnn.304447=US-politics "
    "guardian.221679=GB-politics scotsman.133765=GB-sport scotsman.133804=GB-entertainment guardian.221762=OTHER-politics-BO cnbc.com.6769=US-politics "
    "dailymail.co.uk.298692=EU-entertainment scotsman.133744=GB-world reuters.218911=US-sport newsweek.51310=US-politics independent.226301=US-sport reuters.218923=EU-sport "
    "reuters.218861=US-politics dailymail.co.uk.298759=US-world scotsman.133791=GB-sport cbsnews.248484=EU-scitech dailymail.co.uk.298630=US-scitech "
    "newsweek.51329=US-entertainment bbc.381701=GB-crime dailymail.co.uk.298738=GB-entertainment bbc.381669=OTHER-world-CN foxnews.94512=US-politics "
    "guardian.221718=GB-entertainment dailymail.co.uk.298686=GB-politics cbsnews.248471=US-politics newsweek.51318=US-entertainment rt.com.91335=US-politics "
    "newsweek.51300=US-politics cnn.304478=US-politics upi.176275=US-politics telegraph.405422=OTHER-world-ID reuters.218933=US-politics newsweek.51328=US-politics "
    "newsweek.51307=US-business bbc.381692=GB-world independent.226346=GB-entertainment bbc.381646=GB-sport reuters.218914=US-sport scotsman.133758=EU-sport "
    "rt.com.91350=EU-world scotsman.133773=GB-scitech rt.com.91334=EU-crime bbc.381680=GB-politics guardian.221756=US-politics scotsman.133783=GB-politics cnn.304521=US-sport "
    "dailymail.co.uk.298622=GB-politics bbc.381789=GB-sport dailymail.co.uk.298644=GB-business dailymail.co.uk.298602=GB-world scotsman.133753=GB-sport "
    "independent.226317=GB-entertainment nytimes.184862=US-politics thelocal.65969=OTHER-world-SY nytimes.184825=US-politics cnbc.com.6784=US-politics nytimes.184804=US-politics "
    "nytimes.184830=US-politics scotsman.133801=GB-sport cnbc.com.6770=US-business bbc.381760=GB-crime reuters.218865=OTHER-world-ID newsweek.51339=US-crime "
    "euronews-en.153797=OTHER-world-ID abcnews.306774=US-crime dailymail.co.uk.298696=GB-politics abcnews.306755=US-politics reuters.218909=US-crime "
    "independent.226349=OTHER-sport-RU newsweek.51330=US-politics bbc.381705=GB-sport newsweek.51340=OTHER-world-ID cbsnews.248411=OTHER-world-FM abcnews.306776=US-crime "
    "bbc.381694=GB-entertainment rt.com.91356=US-world telegraph.405430=GB-entertainment telegraph.405404=EU-world bbc.381749=GB-world telegraph.405413=US-politics "
    "bbc.381736=OTHER-politics-KP cbsnews.248394=US-politics nytimes.184822=US-world telegraph.405408=US-politics euronews-en.153799=OTHER-politics-SY "
    "euronews-en.153826=EU-sport cnn.304400=US-world",
}
SUBSETS = {
    k: {d.split("=")[0]: d.split("=")[1] for d in v.split()}
    for (k, v) in SUBSETS.items()
}
COUNTRIES = sorted(list({v.split("-")[0] for v in SUBSETS["wmt19"].values()}))
DOMAINS = sorted(list({v.split("-")[1] for v in SUBSETS["wmt19"].values()}))


def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT.

    :param line: a segment to tokenize
    :return: the tokenized line
    """

    norm = line

    # language-independent part:
    norm = norm.replace("<skipped>", "")
    norm = norm.replace("-\n", "")
    norm = norm.replace("\n", " ")
    norm = norm.replace("&quot;", '"')
    norm = norm.replace("&amp;", "&")
    norm = norm.replace("&lt;", "<")
    norm = norm.replace("&gt;", ">")

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", " \\1 ", norm)
    norm = re.sub(
        r"([^0-9])([\.,])", "\\1 \\2 ", norm
    )  # tokenize period and comma unless preceded by a digit
    norm = re.sub(
        r"([\.,])([^0-9])", " \\1 \\2", norm
    )  # tokenize period and comma unless followed by a digit
    norm = re.sub(
        r"([0-9])(-)", "\\1 \\2 ", norm
    )  # tokenize dash when preceded by a digit
    norm = re.sub(r"\s+", " ", norm)  # one space only between words
    norm = re.sub(r"^\s+", "", norm)  # no leading space
    norm = re.sub(r"\s+$", "", norm)  # no trailing space

    return norm


class UnicodeRegex:
    """Ad-hoc hack to recognize all punctuation and symbols.

    without depending on https://pypi.python.org/pypi/regex/."""

    @staticmethod
    def _property_chars(prefix):
        return "".join(
            chr(x)
            for x in range(sys.maxunicode)
            if unicodedata.category(chr(x)).startswith(prefix)
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punctuation():
        return UnicodeRegex._property_chars("P")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def nondigit_punct_re():
        return re.compile(r"([^\d])([" + UnicodeRegex.punctuation() + r"])")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punct_nondigit_re():
        return re.compile(r"([" + UnicodeRegex.punctuation() + r"])([^\d])")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def symbol_re():
        return re.compile("([" + UnicodeRegex._property_chars("S") + "])")


def tokenize_v14_international(string):
    r"""Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g., a year) followed by a dot at the end of sentence is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).

    :param string: the input string
    :return: a list of tokens
    """
    string = UnicodeRegex.nondigit_punct_re().sub(r"\1 \2 ", string)
    string = UnicodeRegex.punct_nondigit_re().sub(r" \1 \2", string)
    string = UnicodeRegex.symbol_re().sub(r" \1 ", string)
    return string.strip()


def tokenize_zh(sentence):
    """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    The tokenization of Chinese text in this script contains two steps: separate each Chinese
    characters (by utf-8 encoding); tokenize the non Chinese part (following the mteval script).
    Author: Shujian Huang huangsj@nju.edu.cn

    :param sentence: input sentence
    :return: tokenized sentence
    """

    def is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        if (
            uchar >= u"\u3400" and uchar <= u"\u4db5"
        ):  # CJK Unified Ideographs Extension A, release 3.0
            return True
        elif (
            uchar >= u"\u4e00" and uchar <= u"\u9fa5"
        ):  # CJK Unified Ideographs, release 1.1
            return True
        elif (
            uchar >= u"\u9fa6" and uchar <= u"\u9fbb"
        ):  # CJK Unified Ideographs, release 4.1
            return True
        elif (
            uchar >= u"\uf900" and uchar <= u"\ufa2d"
        ):  # CJK Compatibility Ideographs, release 1.1
            return True
        elif (
            uchar >= u"\ufa30" and uchar <= u"\ufa6a"
        ):  # CJK Compatibility Ideographs, release 3.2
            return True
        elif (
            uchar >= u"\ufa70" and uchar <= u"\ufad9"
        ):  # CJK Compatibility Ideographs, release 4.1
            return True
        elif (
            uchar >= u"\u20000" and uchar <= u"\u2a6d6"
        ):  # CJK Unified Ideographs Extension B, release 3.1
            return True
        elif (
            uchar >= u"\u2f800" and uchar <= u"\u2fa1d"
        ):  # CJK Compatibility Supplement, release 3.1
            return True
        elif (
            uchar >= u"\uff00" and uchar <= u"\uffef"
        ):  # Full width ASCII, full width of English punctuation, half width Katakana, half wide half width kana, Korean alphabet
            return True
        elif uchar >= u"\u2e80" and uchar <= u"\u2eff":  # CJK Radicals Supplement
            return True
        elif uchar >= u"\u3000" and uchar <= u"\u303f":  # CJK punctuation mark
            return True
        elif uchar >= u"\u31c0" and uchar <= u"\u31ef":  # CJK stroke
            return True
        elif uchar >= u"\u2f00" and uchar <= u"\u2fdf":  # Kangxi Radicals
            return True
        elif uchar >= u"\u2ff0" and uchar <= u"\u2fff":  # Chinese character structure
            return True
        elif uchar >= u"\u3100" and uchar <= u"\u312f":  # Phonetic symbols
            return True
        elif (
            uchar >= u"\u31a0" and uchar <= u"\u31bf"
        ):  # Phonetic symbols (Taiwanese and Hakka expansion)
            return True
        elif uchar >= u"\ufe10" and uchar <= u"\ufe1f":
            return True
        elif uchar >= u"\ufe30" and uchar <= u"\ufe4f":
            return True
        elif uchar >= u"\u2600" and uchar <= u"\u26ff":
            return True
        elif uchar >= u"\u2700" and uchar <= u"\u27bf":
            return True
        elif uchar >= u"\u3200" and uchar <= u"\u32ff":
            return True
        elif uchar >= u"\u3300" and uchar <= u"\u33ff":
            return True

        return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for char in sentence:
        if is_chinese_char(char):
            sentence_in_chars += " "
            sentence_in_chars += char
            sentence_in_chars += " "
        else:
            sentence_in_chars += char
    sentence = sentence_in_chars

    # TODO: the code above could probably be replaced with the following line:
    # import regex
    # sentence = regex.sub(r'(\p{Han})', r' \1 ', sentence)

    # tokenize punctuation
    sentence = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", sentence)

    # tokenize period and comma unless preceded by a digit
    sentence = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", sentence)

    # tokenize period and comma unless followed by a digit
    sentence = re.sub(r"([\.,])([^0-9])", r" \1 \2", sentence)

    # tokenize dash when preceded by a digit
    sentence = re.sub(r"([0-9])(-)", r"\1 \2 ", sentence)

    # one space only between words
    sentence = re.sub(r"\s+", r" ", sentence)

    # no leading or trailing spaces
    sentence = sentence.strip()

    return sentence


TOKENIZERS = {
    "13a": tokenize_13a,
    "intl": tokenize_v14_international,
    "zh": tokenize_zh,
    "none": lambda x: x,
}
DEFAULT_TOKENIZER = "13a"


def smart_open(file, mode="rt", encoding="utf-8"):
    """Convenience function for reading compressed or plain text files.
    :param file: The file to read.
    :param mode: The file mode (read, write).
    :param encoding: The file encoding.
    """
    if file.endswith(".gz"):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)


def bleu_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the signature
    """

    # Abbreviations for the signature
    abbr = {
        "test": "t",
        "lang": "l",
        "smooth": "s",
        "case": "c",
        "tok": "tok",
        "numrefs": "#",
        "version": "v",
        "origlang": "o",
        "subset": "S",
    }

    signature = {
        "tok": args.tokenize,
        "version": VERSION,
        "smooth": args.smooth,
        "numrefs": numrefs,
        "case": "lc" if args.lc else "mixed",
    }

    if args.test_set is not None:
        signature["test"] = args.test_set

    if args.langpair is not None:
        signature["lang"] = args.langpair

    if args.origlang is not None:
        signature["origlang"] = args.origlang
    if args.subset is not None:
        signature["subset"] = args.subset

    sigstr = "+".join(
        [
            "{}.{}".format(abbr[x] if args.short else x, signature[x])
            for x in sorted(signature.keys())
        ]
    )

    return sigstr


def chrf_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the chrF signature
    """

    # Abbreviations for the signature
    abbr = {
        "test": "t",
        "lang": "l",
        "numchars": "n",
        "space": "s",
        "case": "c",
        "numrefs": "#",
        "version": "v",
        "origlang": "o",
        "subset": "S",
    }

    signature = {
        "version": VERSION,
        "space": args.chrf_whitespace,
        "numchars": args.chrf_order,
        "numrefs": numrefs,
        "case": "lc" if args.lc else "mixed",
    }

    if args.test_set is not None:
        signature["test"] = args.test_set

    if args.langpair is not None:
        signature["lang"] = args.langpair

    if args.origlang is not None:
        signature["origlang"] = args.origlang
    if args.subset is not None:
        signature["subset"] = args.subset

    sigstr = "+".join(
        [
            "{}.{}".format(abbr[x] if args.short else x, signature[x])
            for x in sorted(signature.keys())
        ]
    )

    return sigstr


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

    :param line: A segment containing a sequence of words.
    :param min_order: Minimum n-gram length (default: 1).
    :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1

    return ngrams


def extract_char_ngrams(s: str, n: int) -> Counter:
    """
    Yields counts of character n-grams from string s of order n.
    """
    return Counter([s[i : i + n] for i in range(len(s) - n + 1)])


def ref_stats(output, refs):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs:
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len


def _clean(s):
    """
    Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.

    :param s: The string.
    :return: A cleaned-up string.
    """
    return re.sub(r"\s+", " ", s.strip())


def process_to_text(rawfile, txtfile, field: int = None):
    """Processes raw files to plain text files.
    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    :param field: For TSV files, which field to extract.
    """

    if not os.path.exists(txtfile) or os.path.getsize(txtfile) == 0:
        if rawfile.endswith(".sgm") or rawfile.endswith(".sgml"):
            with smart_open(rawfile) as fin, smart_open(txtfile, "wt") as fout:
                for line in fin:
                    if line.startswith("<seg "):
                        print(
                            _clean(re.sub(r"<seg.*?>(.*)</seg>.*?", "\\1", line)),
                            file=fout,
                        )
        elif rawfile.endswith(".xml"):  # IWSLT
            with smart_open(rawfile) as fin, smart_open(txtfile, "wt") as fout:
                for line in fin:
                    if line.startswith("<seg "):
                        print(
                            _clean(re.sub(r"<seg.*?>(.*)</seg>.*?", "\\1", line)),
                            file=fout,
                        )
        elif rawfile.endswith(".txt"):  # wmt17/ms
            with smart_open(rawfile) as fin, smart_open(txtfile, "wt") as fout:
                for line in fin:
                    print(line.rstrip(), file=fout)
        elif rawfile.endswith(".tsv"):  # MTNT
            with smart_open(rawfile) as fin, smart_open(txtfile, "wt") as fout:
                for line in fin:
                    print(line.rstrip().split("\t")[field], file=fout)


class Result:
    def __init__(self, score: float):
        self.score = score

    def __str__(self):
        return self.format()


class BLEU:
    def __init__(self, scores, counts, totals, precisions, bp, sys_len, ref_len):

        self.scores = scores
        self.counts = counts
        self.totals = totals
        self.precisions = precisions
        self.bp = bp
        self.sys_len = sys_len
        self.ref_len = ref_len

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in self.precisions])
        return "BLEU = {scores} {precisions} (BP = {bp:.3f} ratio = {ratio:.3f} hyp_len = {sys_len:d} ref_len = {ref_len:d})".format(
            scores=self.scores,
            width=width,
            precisions=precisions,
            bp=self.bp,
            ratio=self.sys_len / self.ref_len,
            sys_len=self.sys_len,
            ref_len=self.ref_len,
        )


class CHRF(Result):
    def __init__(self, score: float):
        super().__init__(score)

    def format(self, width=2):
        return "{score:.{width}f}".format(score=self.score, width=width)


def compute_bleu(
    correct: List[int],
    total: List[int],
    sys_len: int,
    ref_len: int,
    smooth_method="none",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    use_effective_order=False,
) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
    Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
    :return: A BLEU object with the score (100-based) and other statistics.
    """

    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.0
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if smooth_method == "add-k" and n > 1:
            correct[n] += smooth_value
            total[n] += smooth_value
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1

        if correct[n] == 0:
            if smooth_method == "exp":
                smooth_mteval *= 2
                precisions[n] = 100.0 / (smooth_mteval * total[n])
            elif smooth_method == "floor":
                precisions[n] = 100.0 * smooth_value / total[n]
        else:
            precisions[n] = 100.0 * correct[n] / total[n]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
    # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
    # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
    # maximum order. It is only available through the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    scores = []
    for effective_order in range(1, NGRAM_ORDER + 1):
        scores.append(
            brevity_penalty
            * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)
        )

    return BLEU(scores, correct, total, precisions, brevity_penalty, sys_len, ref_len)


def sentence_bleu(
    hypothesis: str,
    references: List[str],
    smooth_method: str = "floor",
    smooth_value: float = SMOOTH_VALUE_DEFAULT,
    use_effective_order: bool = True,
) -> BLEU:
    """
    Computes BLEU on a single sentence pair.

    Disclaimer: computing BLEU on the sentence level is not its intended use,
    BLEU is a corpus-level metric.

    :param hypothesis: Hypothesis string.
    :param reference: Reference string.
    :param smooth_value: For 'floor' smoothing, the floor value to use.
    :param use_effective_order: Account for references that are shorter than the largest n-gram.
    :return: Returns a single BLEU score as a float.
    """
    bleu = corpus_bleu(
        hypothesis,
        references,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )
    return bleu


def corpus_bleu(
    sys_stream: Union[str, Iterable[str]],
    ref_streams: Union[str, List[Iterable[str]]],
    smooth_method="exp",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    force=False,
    lowercase=False,
    tokenize=DEFAULT_TOKENIZER,
    use_effective_order=False,
) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_value: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """

    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]
    total = [0 for n in range(NGRAM_ORDER)]

    # look for already-tokenized sentences
    tokenized_count = 0

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == "none") and lines[0].rstrip().endswith(" ."):
            tokenized_count += 1

        output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n - 1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n - 1] += sys_ngrams[ngram]

    return compute_bleu(
        correct,
        total,
        sys_len,
        ref_len,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )


def raw_corpus_bleu(sys_stream, ref_streams, smooth_value=SMOOTH_VALUE_DEFAULT) -> BLEU:
    """Convenience function that wraps corpus_bleu().
    This is convenient if you're using sacrebleu as a library, say for scoring on dev.
    It uses no tokenization and 'floor' smoothing, with the floor default to 0 (no smoothing).

    :param sys_stream: the system stream (a sequence of segments)
    :param ref_streams: a list of one or more reference streams (each a sequence of segments)
    """
    return corpus_bleu(
        sys_stream,
        ref_streams,
        smooth_method="floor",
        smooth_value=smooth_value,
        force=True,
        tokenize="none",
        use_effective_order=True,
    )


def delete_whitespace(text: str) -> str:
    """
    Removes whitespaces from text.
    """
    return re.sub(r"\s+", "", text).strip()


def get_sentence_statistics(
    hypothesis: str,
    reference: str,
    order: int = CHRF_ORDER,
    remove_whitespace: bool = True,
) -> List[float]:
    hypothesis = delete_whitespace(hypothesis) if remove_whitespace else hypothesis
    reference = delete_whitespace(reference) if remove_whitespace else reference
    statistics = [0] * (order * 3)
    for i in range(order):
        n = i + 1
        hypothesis_ngrams = extract_char_ngrams(hypothesis, n)
        reference_ngrams = extract_char_ngrams(reference, n)
        common_ngrams = hypothesis_ngrams & reference_ngrams
        statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
        statistics[3 * i + 1] = sum(reference_ngrams.values())
        statistics[3 * i + 2] = sum(common_ngrams.values())
    return statistics


def get_corpus_statistics(
    hypotheses: Iterable[str],
    references: Iterable[str],
    order: int = CHRF_ORDER,
    remove_whitespace: bool = True,
) -> List[float]:
    corpus_statistics = [0] * (order * 3)
    for hypothesis, reference in zip(hypotheses, references):
        statistics = get_sentence_statistics(
            hypothesis, reference, order=order, remove_whitespace=remove_whitespace
        )
        for i in range(len(statistics)):
            corpus_statistics[i] += statistics[i]
    return corpus_statistics


def _avg_precision_and_recall(
    statistics: List[float], order: int
) -> Tuple[float, float]:
    avg_precision = 0.0
    avg_recall = 0.0
    effective_order = 0
    for i in range(order):
        hypotheses_ngrams = statistics[3 * i + 0]
        references_ngrams = statistics[3 * i + 1]
        common_ngrams = statistics[3 * i + 2]
        if hypotheses_ngrams > 0 and references_ngrams > 0:
            avg_precision += common_ngrams / hypotheses_ngrams
            avg_recall += common_ngrams / references_ngrams
            effective_order += 1
    if effective_order == 0:
        return 0.0, 0.0
    avg_precision /= effective_order
    avg_recall /= effective_order
    return avg_precision, avg_recall


def _chrf(avg_precision, avg_recall, beta: int = CHRF_BETA) -> float:
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_square = beta ** 2
    score = (
        (1 + beta_square)
        * (avg_precision * avg_recall)
        / ((beta_square * avg_precision) + avg_recall)
    )
    return score


def corpus_chrf(
    hypotheses: Iterable[str],
    references: Iterable[str],
    order: int = CHRF_ORDER,
    beta: float = CHRF_BETA,
    remove_whitespace: bool = True,
) -> CHRF:
    """
    Computes Chrf on a corpus.

    :param hypotheses: Stream of hypotheses.
    :param references: Stream of references
    :param order: Maximum n-gram order.
    :param remove_whitespace: Whether to delete all whitespace from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    corpus_statistics = get_corpus_statistics(
        hypotheses, references, order=order, remove_whitespace=remove_whitespace
    )
    avg_precision, avg_recall = _avg_precision_and_recall(corpus_statistics, order)
    return CHRF(_chrf(avg_precision, avg_recall, beta=beta))


def sentence_chrf(
    hypothesis: str,
    reference: str,
    order: int = CHRF_ORDER,
    beta: float = CHRF_BETA,
    remove_whitespace: bool = True,
) -> CHRF:
    """
    Computes ChrF on a single sentence pair.

    :param hypothesis: Hypothesis string.
    :param reference: Reference string.
    :param order: Maximum n-gram order.
    :param remove_whitespace: Whether to delete whitespaces from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    statistics = get_sentence_statistics(
        hypothesis, reference, order=order, remove_whitespace=remove_whitespace
    )
    avg_precision, avg_recall = _avg_precision_and_recall(statistics, order)
    return CHRF(_chrf(avg_precision, avg_recall, beta=beta))

def main():
    arg_parser = argparse.ArgumentParser(
        description="sacreBLEU: Hassle-free computation of shareable BLEU scores.\n"
        "Quick usage: score your detokenized output against WMT'14 EN-DE:\n"
        "    cat output.detok.de | sacrebleu -t wmt14 -l en-de",
        # epilog = 'Available test sets: ' + ','.join(sorted(DATASETS.keys(), reverse=True)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "--test-set",
        "-t",
        type=str,
        default=None,
        help="the test set to use (see also --list) or a comma-separated list of test sets to be concatenated",
    )
    arg_parser.add_argument(
        "-lc",
        action="store_true",
        default=False,
        help="Use case-insensitive BLEU (default: actual case)",
    )
    arg_parser.add_argument(
        "--sentence-level",
        "-sl",
        action="store_true",
        help="Output metric on each sentence.",
    )
    arg_parser.add_argument(
        "--smooth",
        "-s",
        choices=["exp", "floor", "add-n", "none"],
        default="exp",
        help="smoothing method: exponential decay (default), floor (increment zero counts), add-k (increment num/denom by k for n>1), or none",
    )
    arg_parser.add_argument(
        "--smooth-value",
        "-sv",
        type=float,
        default=SMOOTH_VALUE_DEFAULT,
        help="The value to pass to the smoothing technique, when relevant. Default: %(default)s.",
    )
    arg_parser.add_argument(
        "--tokenize",
        "-tok",
        choices=TOKENIZERS.keys(),
        default=None,
        help="tokenization method to use",
    )
    arg_parser.add_argument(
        "--language-pair",
        "-l",
        dest="langpair",
        default=None,
        help="source-target language pair (2-char ISO639-1 codes)",
    )
    arg_parser.add_argument(
        "--origlang",
        "-ol",
        dest="origlang",
        default=None,
        help='use a subset of sentences with a given original language (2-char ISO639-1 codes), "non-" prefix means negation',
    )
    arg_parser.add_argument(
        "--subset",
        dest="subset",
        default=None,
        help="use a subset of sentences whose document annotation matches a give regex (see SUBSETS in the source code)",
    )
    arg_parser.add_argument(
        "--download", type=str, default=None, help="download a test set and quit"
    )
    arg_parser.add_argument(
        "--echo",
        choices=["src", "ref", "both"],
        type=str,
        default=None,
        help="output the source (src), reference (ref), or both (both, pasted) to STDOUT and quit",
    )
    arg_parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="-",
        help="Read input from a file instead of STDIN",
    )
    arg_parser.add_argument(
        "--num-refs",
        "-nr",
        type=int,
        default=1,
        help="Split the reference stream on tabs, and expect this many references. Default: %(default)s.",
    )
    arg_parser.add_argument(
        "refs",
        nargs="*",
        default=[],
        help="optional list of references (for backwards-compatibility with older scripts)",
    )
    arg_parser.add_argument(
        "--metrics",
        "-m",
        choices=["bleu", "chrf"],
        nargs="+",
        default=["bleu"],
        help="metrics to compute (default: bleu)",
    )
    arg_parser.add_argument(
        "--chrf-order",
        type=int,
        default=CHRF_ORDER,
        help="chrf character order (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--chrf-beta",
        type=int,
        default=CHRF_BETA,
        help="chrf BETA parameter (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--chrf-whitespace",
        action="store_true",
        default=False,
        help="include whitespace in chrF calculation (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--short",
        default=False,
        action="store_true",
        help="produce a shorter (less human readable) signature",
    )
    arg_parser.add_argument(
        "--score-only",
        "-b",
        default=False,
        action="store_true",
        help="output only the BLEU score",
    )
    arg_parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="insist that your tokenized input is actually detokenized",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        default=False,
        action="store_true",
        help="suppress informative output",
    )
    arg_parser.add_argument(
        "--encoding",
        "-e",
        type=str,
        default="utf-8",
        help="open text files with specified encoding (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--list",
        default=False,
        action="store_true",
        help="print a list of all available test sets.",
    )
    arg_parser.add_argument(
        "--citation",
        "--cite",
        default=False,
        action="store_true",
        help="dump the bibtex citation and quit.",
    )
    arg_parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=1,
        help="floating point width (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--detail",
        "-d",
        default=False,
        action="store_true",
        help="print extra information (split test sets based on origlang)",
    )
    arg_parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s {}".format(VERSION)
    )
    args = arg_parser.parse_args()

    # Explicitly set the encoding
    sys.stdin = open(
        sys.stdin.fileno(), mode="r", encoding="utf-8", buffering=True, newline="\n"
    )
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=True)

    # Internal tokenizer settings. Set to 'zh' for Chinese  DEFAULT_TOKENIZER (
    if args.tokenize is None:
        # set default
        if args.langpair is not None and args.langpair.split("-")[1] == "zh":
            args.tokenize = "zh"
        else:
            args.tokenize = DEFAULT_TOKENIZER

    # concat_ref_files is a list of list of reference filenames, for example:
    # concat_ref_files = [[testset1_refA, testset1_refB], [testset2_refA, testset2_refB]]
    if args.test_set is None:
        concat_ref_files = [args.refs]

    inputfh = (
        io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
        if args.input == "-"
        else smart_open(args.input, encoding=args.encoding)
    )
    full_system = inputfh.readlines()

    # Read references
    full_refs = [[] for x in range(max(len(concat_ref_files[0]), args.num_refs))]
    for ref_files in concat_ref_files:
        for refno, ref_file in enumerate(ref_files):
            for lineno, line in enumerate(
                smart_open(ref_file, encoding=args.encoding), 1
            ):
                if args.num_refs != 1:
                    splits = line.rstrip().split(sep="\t", maxsplit=args.num_refs - 1)
                else:
                    full_refs[refno].append(line)


    # Handle sentence level and quit

    # Else, handle system level
    results = []


def display_metric(metrics_to_print, results, num_refs, args):
    """
    Badly in need of refactoring.
    One idea is to put all of this in the BLEU and CHRF classes, and then define
    a Result::signature() function.
    """
    for metric, result in zip(metrics_to_print, results):
        if metric == "bleu":
            if args.score_only:
                print("{0:.{1}f}".format(result.score, args.width))
            else:
                version_str = bleu_signature(args, num_refs)
                print(result.format(args.width).replace("BLEU", "BLEU+" + version_str))

        elif metric == "chrf":
            if args.score_only:
                print("{0:.{1}f}".format(result.score, args.width))
            else:
                version_str = chrf_signature(args, num_refs)
                print(
                    "chrF{0:d}+{1} = {2:.{3}f}".format(
                        args.chrf_beta, version_str, result.score, args.width
                    )
                )


if __name__ == "__main__":
    main()
