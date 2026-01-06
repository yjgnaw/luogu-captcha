<?php

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;
use Gregwar\Captcha\PhraseBuilder;

// Create output directory if it doesn't exist
$outputDir = __DIR__ . '/captchas';
if (!is_dir($outputDir)) {
    mkdir($outputDir, 0755, true);
}

// Parse command-line arguments
$count = 50000; // Default count
if (isset($argv[1])) {
    $count = (int) $argv[1];
    if ($count <= 0) {
        echo "Error: Count must be a positive integer.\n";
        exit(1);
    }
}

// Open labels file
$labelsFile = fopen($outputDir . '/labels.csv', 'w');

echo "Generating $count 4-character captchas...\n";

$phraseBuilder = new PhraseBuilder(4);

for ($i = 1; $i <= $count; $i++) {
    $builder = new CaptchaBuilder(null, $phraseBuilder);
    // Luogu captchas are 90x35
    $builder->build(90, 35);

    $filename = sprintf('captcha_%05d.jpg', $i);
    $builder->save($outputDir . '/' . $filename);

    // Save the label (phrase)
    $phrase = $builder->getPhrase();
    fwrite($labelsFile, "$filename,$phrase\n");

    if ($i % 500 == 0) {
        echo "Generated $i captchas...\n";
    }
}

fclose($labelsFile);

echo "Done! Generated $count 4-character captchas in the 'captchas' directory.\n";
echo "Labels saved to captchas/labels.csv\n";
?>