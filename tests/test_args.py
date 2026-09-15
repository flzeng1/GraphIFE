import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import yaml

from args import parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config.yaml'


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / 'custom.yaml'

    def parse_config(self, values, *argv):
        self.config.write_text(yaml.safe_dump(values))
        return parse_args(['--config', str(self.config), *argv])

    def assert_parse_error(self, argv, message):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
            parse_args(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, error.getvalue())

    def test_public_parameters_match_yaml(self):
        configured = yaml.safe_load(CONFIG.read_text())
        args = parse_args(['--config', str(CONFIG)])
        self.assertEqual(len(configured), 27)
        self.assertEqual(set(vars(args)), set(configured))
        for key, value in configured.items():
            if key != 'config':
                expected = float(value) if key == 'lr_min' else value
                self.assertEqual(getattr(args, key), expected, key)

    def test_custom_yaml_and_command_line_precedence(self):
        args = self.parse_config(
            {'dataset': 'Cora', 'net': 'GCN', 'alpha': 0.35, 'warmup': 10},
            '--net', 'SAGE', '--alpha', '0.4')
        self.assertEqual(args.dataset, 'Cora')
        self.assertEqual(args.net, 'SAGE')
        self.assertEqual(args.alpha, 0.4)
        self.assertEqual(args.warmup, 10)
        self.assertEqual(args.config, str(self.config))

    def test_boolean_options(self):
        for name in ('early_stop', 'ife', 'verbose', 'write'):
            for text, expected in [('false', False), ('true', True)]:
                with self.subTest(name=name, text=text):
                    args = self.parse_config({name: not expected}, '--' + name, text)
                    self.assertIs(getattr(args, name), expected)
        self.assertIs(self.parse_config({'ife': 'false'}).ife, False)

    def test_unknown_yaml_fields_cannot_be_injected(self):
        for key in ('drop_edge', 'beta_alpha', 'synthetic_pair_penalty_mode', 'alhpa', 'help'):
            with self.subTest(key=key):
                self.config.write_text(yaml.safe_dump({key: 0.1}))
                self.assert_parse_error(['--config', str(self.config)], 'Unknown YAML parameter')

    def test_removed_and_abbreviated_options_are_rejected(self):
        for option in ('--drop_edge', '--beta_alpha', '--alph'):
            with self.subTest(option=option):
                self.assert_parse_error(['--config', str(CONFIG), option, '0.2'], 'unrecognized arguments')

    def test_invalid_yaml_values(self):
        cases = [{'dataset': 'Unknown'}, {'epochs': 2.5}, {'epochs': True},
                 {'ife': 'maybe'}, {'alpha': None}, {'alpha': float('nan')},
                 {'data_path': ['data']}]
        for values in cases:
            with self.subTest(values=values):
                self.config.write_text(yaml.safe_dump(values))
                self.assert_parse_error(['--config', str(self.config)], 'Invalid YAML value')
        self.config.write_text('- Cora\n- GCN\n')
        self.assert_parse_error(['--config', str(self.config)], 'must contain a mapping')

    def test_empty_yaml_and_scientific_notation(self):
        self.config.write_text('')
        args = parse_args(['--config', str(self.config)])
        self.assertEqual(len(vars(args)), 27)
        self.assertEqual(self.parse_config({'lr_min': '1e-06'}).lr_min, 1e-6)

    def test_help_does_not_require_existing_yaml(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            parse_args(['--config', str(self.config), '--help'])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('--config', output.getvalue())
        self.assertNotIn('--drop_edge', output.getvalue())


if __name__ == '__main__':
    unittest.main()
