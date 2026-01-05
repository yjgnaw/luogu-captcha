<?php

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;
use Gregwar\Captcha\PhraseBuilder;

// Create output directory if it doesn't exist
$outputDir = __DIR__ . '/captchas';
if (!is_dir($outputDir)) {
    mkdir($outputDir, 0755, true);
}

// Create a custom phrase builder that generates 4-character phrases
class FourCharPhraseBuilder extends PhraseBuilder
{
    public function build($length = null, $charset = null)
    {
        $phrase = '';
        $charset = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz';
        for ($i = 0; $i < 4; $i++) {
            $phrase .= $charset[rand(0, strlen($charset) - 1)];
        }
        return $phrase;
    }
}

// Open labels file
$labelsFile = fopen($outputDir . '/labels.csv', 'w');

echo "Generating 10000 4-character captchas...\n";

$phraseBuilder = new FourCharPhraseBuilder();

for ($i = 1; $i <= 10000; $i++) {
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

echo "Done! Generated 10000 4-character captchas in the 'captchas_4char' directory.\n";
echo "Labels saved to captchas/labels.csv\n";
?>