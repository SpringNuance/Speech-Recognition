#!/usr/bin/perl

use locale;
use strict;

my $proto_file = shift;
my $phoneme_list_file = shift;
my $target_path = shift;

if (!(-e $proto_file) || !(-e $phoneme_list_file)) {
    print "Usage: proto2hmms.pl <proto-hmm> <phoneme-list> [target-path]\n";
    exit(1);
}

$target_path = "." if (length($target_path) < 1);

my ($fh, $phoneme_fh);
open($fh, "< $proto_file") || die("Could not open $proto_file");
my @proto_hmm = <$fh>;
close($fh);

open($phoneme_fh, "< $phoneme_list_file") || 
    die("Could not open $phoneme_list_file");

while (<$phoneme_fh>) {
    chomp;
    my $phoneme = $_;
    my @hmm = @proto_hmm;
    $hmm[0] =~ s/proto/$phoneme/;
    my $target_file_name = $target_path."/".$phoneme;
    open($fh, "> $target_file_name") || die("Could not open $target_file_name");
    print $fh @hmm;
    close($fh);
}

close($phoneme_fh);
