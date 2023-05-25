TESTS = [
    [
        """$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k""",
        "24933642",
    ],
    [
        """$ cd /
$ ls
dir a
8504156 c.dat
dir d
$ cd a
$ ls
dir e
4060174 j
8033020 d.log
5626152 d.ext
7214296 k
$ cd e
$ ls
29116 f
2557 g
62596 h.lst
$ cd ..
$ cd ..
$ cd d
$ ls
14848514 b.txt
584 i""",
        "14849098",
    ],
]
