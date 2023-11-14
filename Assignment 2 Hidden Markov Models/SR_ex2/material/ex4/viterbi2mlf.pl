#!/usr/bin/perl

print "#!MLF!#\n";

while (<>) {
    chomp;
    if (/^(\S+) <s> (.*)<\/s>/) {
	$file = $1;
	$_ = $2;
	print "\"$file\"\n";
	print "<s>\n";
	s/<.*?>//g;
	for (split(/\s+/, $_)) {
	    print "$_\n";
	}
        print "</s>\n";
	print ".\n";
    }
}
