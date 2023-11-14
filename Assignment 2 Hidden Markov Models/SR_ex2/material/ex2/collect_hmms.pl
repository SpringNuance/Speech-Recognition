#!/usr/bin/perl

use locale;
use strict;

my $phoneme_list_file = shift;
my $hmm_path = shift;

if (!(-e $phoneme_list_file)) {
    print "Usage: collect_hmms.pl <phoneme-list> [hmm-path]\n";
    exit(1);
}

$hmm_path = "." if (length($hmm_path) < 1);

my ($fh, $phoneme_fh);
open($phoneme_fh, "< $phoneme_list_file") || 
    die("Could not open $phoneme_list_file");

while (<$phoneme_fh>) {    
    chomp;
    my $phoneme = $_;
    my $hmm_name = $hmm_path."/".$phoneme;
    open($fh, "< $hmm_name") || die("Could not open $hmm_name");
    while (<$fh>) {
	last if (/^~h/);
    }
    print $_;
    print <$fh>;
    close($fh);
}

close($phoneme_fh);
