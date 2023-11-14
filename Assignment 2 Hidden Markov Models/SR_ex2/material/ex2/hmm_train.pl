#!/usr/bin/perl

use locale;
use strict;

my $htk_path = shift;
my $mlf_file = shift;
my $scp_file = shift;
my $phoneme_list = shift;
my $config = shift;

if (!(-e $htk_path) || !(-e $mlf_file) || !(-e $scp_file) ||
    !(-e $phoneme_list) || !(-e $config)) {
    print "Usage: hmm_train.pl <htk-path> <mlf-file> <scp-file> <phoneme-list> <config>\n";
    exit(1);
}

### Settings ###
my $avg_num_gaussians = 8; # Average number of Gaussians per state
my $num_iter = 4; # Number of iterations
my $num_reest = 3; # Number of re-estimations in each iteration
################


### Initial directories ###
my $source_path = "hmm-2";
my $target_path = "hmm-3";
###########################


# Training loop
for (my $iter = 1; $iter <= $num_iter; $iter++) {
    mkdir $target_path;
    print "Iteration $iter\n";

    # Split the Gaussians
    # Write a script for HHEd to split Gaussians in mixtures
    # according to the number of frames available (PS command)
    my $fh;
    open($fh, "> $source_path/split.hed") || die("Could not open $source_path/split.hed for writing");
    print $fh "LS $source_path/stats\n"; # Load statistics
    print $fh "PS $avg_num_gaussians 0.4 ".($num_iter-$iter+1)."\n";
    close($fh);
    system("$htk_path/HHEd -H $source_path/macros -H $source_path/hmmdefs -M $target_path $source_path/split.hed $phoneme_list") && die("HHEd failed");

    # Re-estimation loop
    for (my $estim_iter = 1; $estim_iter <= $num_reest; $estim_iter++) {
	print  "  Re-estimating $estim_iter/$num_reest\n";
	system("$htk_path/HERest -T 1 -C $config -I $mlf_file -t 250.0 150.0 1000.0 -S $scp_file -H $target_path/macros -H $target_path/hmmdefs -s $target_path/stats $phoneme_list > $target_path/train_$estim_iter.log") && die("HERest failed");
    }
    $source_path = $target_path;
    my $new_index = ($iter+3);
    $target_path =~ s/\-\d/\-$new_index/; # Update the target path
}
