import re
import itertools

from difflib import Differ, SequenceMatcher


class POSIXDiffer(Differ):
    """
    This class produces differences in the POSIX default format
    (see http://www.unix.com/man-page/POSIX/1posix/diff/),
    which is the same as the Gnu diff "normal format"
    (see http://www.gnu.org/software/diffutils/manual/diffutils.html#Normal).
    """

    def compare(self, a, b):
        cruncher = SequenceMatcher(self.linejunk, a, b)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if alo == ahi:
                f1 = "%d" % alo
            elif alo + 1 == ahi:
                f1 = "%d" % (alo + 1)
            else:
                f1 = "%d,%d" % (alo + 1, ahi)

            if blo == bhi:
                f2 = "%d" % blo
            elif blo + 1 == bhi:
                f2 = "%d" % (blo + 1)
            else:
                f2 = "%d,%d" % (blo + 1, bhi)

            if tag == "replace":
                g = itertools.chain(
                    ["%sc%s\n" % (f1, f2)],
                    self._my_plain_replace(a, alo, ahi, b, blo, bhi),
                )
            elif tag == "delete":
                g = itertools.chain(
                    ["%sd%s\n" % (f1, f2)], self._dump("<", a, alo, ahi)
                )
            elif tag == "insert":
                g = itertools.chain(
                    ["%sa%s\n" % (f1, f2)], self._dump(">", b, blo, bhi)
                )
            elif tag == "equal":
                g = []
            else:
                raise (ValueError, "unknown tag %r" % (tag,))

            for line in g:
                yield line

    def _my_plain_replace(self, a, alo, ahi, b, blo, bhi):
        assert alo < ahi and blo < bhi
        first = self._dump("<", a, alo, ahi)
        second = self._dump(">", b, blo, bhi)

        for g in first, ["---\n"], second:
            for line in g:
                yield line


def pdiff(a, b):
    """
    Compare `a` and `b` (lists of strings); return a POSIX/Gnu "normal format" diff.
    """
    return POSIXDiffer().compare(a, b)


def group_diff_chunks(diff):
    """
    This function creates a list by extracting the start line, end line for
    the files in the diff that are of the form n[,n]cn[,n] (the output of the
    diff command in shell)
    """

    def normalize_chunk(chunk):
        s1, e1, op, s2, e2 = chunk

        s1 = int(s1)
        e1 = (s1 if not e1 else int(e1)) + 1
        s2 = int(s2)
        e2 = (s2 if not e2 else int(e2)) + 1

        return s1, e1, op, s2, e2

    p = re.compile(r"^([0-9]+),?([0-9]*)([a-zA-Z])([0-9]+),?([0-9]*)", re.MULTILINE)
    chunks = p.findall(diff)

    return [normalize_chunk(chunk) for chunk in chunks]


def single_change(chunk):
    s1, e1, _, s2, e2 = chunk

    return e1 - s1 == e2 - s2 == 1


def single_line_changed(diff):
    chunks = group_diff_chunks(diff)
    return len(chunks) == 1 and single_change(chunks[0])
