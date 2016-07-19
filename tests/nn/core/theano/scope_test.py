from __future__ import absolute_import

import unittest

from luchador.nn.core.theano.scope import (
    _set_flag,
    _get_flag,
    _set_scope,
    _get_scope,
    get_variable,
    VariableScope,
    variable_scope,
    get_variable_scope,
)


def _reset():
    """Reset global state of scope and related objects"""
    import luchador.nn.core.theano.scope as scope_module
    _set_flag(False)
    _set_scope('')
    scope_module._VARIABLES = {}


class TestVariableScopeClass(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', _get_scope()
        # Sanitize global scope for the next test in case something went wrong
        _reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_update_scope(self):
        """VariableScope updates current variable scope"""
        scopes = ['aaa', 'bbb', 'ccc']
        with VariableScope(reuse=False, name=scopes[0]):
            expected = scopes[0]
            found = _get_scope()
            self.assertEqual(
                expected, found,
                'Failed to update current variable scope. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with VariableScope(reuse=False, name=scopes[1]):
                expected = scopes[1]
                found = _get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                with VariableScope(reuse=False, name=scopes[2]):
                    expected = scopes[2]
                    found = _get_scope()
                    self.assertEqual(
                        expected, found,
                        'Failed to update current variable scope. '
                        'Expected: "{}", Found: "{}"'.format(expected, found)
                    )

    def test_update_reuse_flag(self):
        """VariableScope updates current reuse flag"""
        with VariableScope(reuse=False, name='scope'):
            expected = False
            found = _get_flag()
            self.assertEqual(
                expected, found,
                'Failed to update current reuse flag. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with VariableScope(reuse=False, name='scope'):
                expected = False
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
            with VariableScope(reuse=True, name='scope'):
                expected = True
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
        with VariableScope(reuse=True, name='scope'):
            expected = True
            found = _get_flag()
            self.assertEqual(
                expected, found,
                'Failed to update current reuse flag. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with VariableScope(reuse=False, name='scope'):
                expected = False
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
            with VariableScope(reuse=True, name='scope'):
                expected = True
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_reusing_VS_object_correctly_reset_scope(self):
        """Reusing VariableScope changes scope and flag correctly"""
        scope, flag = 'aaa', False
        with VariableScope(reuse=flag, name=scope) as vs:
            pass

        with VariableScope(True, 'scope_to_be_ignored'):
            with vs:
                expected = scope
                found = _get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope '
                    'when reusing VariableScope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                expected = flag
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag '
                    'when reusing VariableScope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_VariableScope_correctly_closes_scope(self):
        """VariableScope revert current variable scope after close"""
        scopes = ['aaa', 'bbb', 'ccc']
        for scope in scopes:
            pre_scope = _get_scope()
            pre_reuse = _get_flag()
            with VariableScope(False, scope):
                pass
            post_scope = _get_scope()
            post_reuse = _get_flag()

            expected = pre_scope
            found = post_scope
            self.assertEqual(
                expected, found,
                'Failed to reset variable scope after closing a scope.'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = pre_reuse
            found = post_reuse
            self.assertEqual(
                expected, found,
                'Failed to reset reuse flag after closing a scope.'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

    def test_reuse_variables_enables_flag(self):
        """reuse_variables call enable reuse flag"""
        with VariableScope(reuse=False, name='base_scope'):
            with VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
                expected = True
                found = _get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope '
                    'with call to reuse_variables. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_flag_reset_after_reuse_variables(self):
        """reuse_variables call enable reuse flag"""
        with VariableScope(reuse=False, name='base_scope'):
            with VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
            expected = False
            found = _get_flag()
            self.assertEqual(
                expected, found,
                'Failed to reset the flag after calling reuse_variables'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        with VariableScope(reuse=True, name='base_scope'):
            with VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
            expected = True
            found = _get_flag()
            self.assertEqual(
                expected, found,
                'Failed to reset the flag after calling reuse_variables'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )


class TestVariableScopeFuncs(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', _get_scope()
        # Sanitize global scope for the next test in case something went wrong
        _reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_variable_scope(self):
        """variable_scope stacks new scope on the current scope"""
        scopes = ['aaa', 'bbb', 'ccc']
        with variable_scope(scopes[0]):
            expected = scopes[0]
            found = _get_scope()
            self.assertEqual(
                expected, found,
                'Failed to update current variable scope with variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with variable_scope(scopes[1]):
                expected = '/'.join(scopes[:2])
                found = _get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to stack variable scopes with variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                with variable_scope(scopes[2]):
                    expected = '/'.join(scopes[:3])
                    found = _get_scope()
                    self.assertEqual(
                        expected, found,
                        'Failed to stack variable scopes with variable_scope'
                        'Expected: "{}", Found: "{}"'.format(expected, found)
                    )
            with variable_scope(scopes[2]):
                expected = '/'.join([scopes[0], scopes[2]])
                found = _get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to stack variable scopes with variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_get_variable_scope(self):
        """get_variable_scope retrieves the current scope and reuse flag"""
        scope, reuse = 'aaa', False
        with variable_scope(scope, reuse=reuse):
            vs = get_variable_scope()

            expected = scope
            found = vs.name
            self.assertEqual(
                expected, found,
                'Failed to get the current scope with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = reuse
            found = vs.reuse
            self.assertEqual(
                expected, found,
                'Failed to get the current reuse flag with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        scope, reuse = 'aaa', True
        with variable_scope(scope, reuse=reuse):
            vs = get_variable_scope()

            expected = scope
            found = vs.name
            self.assertEqual(
                expected, found,
                'Failed to get the current scope with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = reuse
            found = vs.reuse
            self.assertEqual(
                expected, found,
                'Failed to get the current reuse flag with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        scopes, reuses = ['aaa', 'bbb'], [False, False]
        with variable_scope(scopes[0], reuse=reuses[0]):
            with variable_scope(scopes[1], reuse=reuses[1]):
                vs = get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [False, True]
        with variable_scope(scopes[0], reuse=reuses[0]):
            with variable_scope(scopes[1], reuse=reuses[1]):
                vs = get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [True, True]
        with variable_scope(scopes[0], reuse=reuses[0]):
            with variable_scope(scopes[1], reuse=reuses[1]):
                vs = get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [True, False]
        with variable_scope(scopes[0], reuse=reuses[0]):
            with variable_scope(scopes[1], reuse=reuses[1]):
                vs = get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )


class TestGetVariable(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', _get_scope()
        # Sanitize global scope for the next test in case something went wrong
        _reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_get_variable_creates_variable(self):
        """get_variable create variable"""
        get_variable('test_var', shape=[3, 1])

    def test_get_variable_reuse_variable(self):
        """get_variable create variable"""
        name = 'test_var'
        var1 = get_variable(name, shape=[3, 1])
        _set_flag(True)
        var2 = get_variable(name)
        self.assertIs(
            var1, var2,
            'Reused variable should be identical to the original variable')

    def test_get_variable_raises_when_reuseing_non_existent_variable(self):
        """get_variable raise when trying to reuse non existent variable"""
        _set_flag(True)
        try:
            get_variable('non_existent_var')
        except ValueError:
            pass
        else:
            self.fail('get_variable should raise when '
                      'trying to reuse non existent variable.')

    def test_get_variable_raises_when_creating_already_existing_variable(self):
        """get_variable raise when trying to create existent variable"""
        name = 'aaa'
        get_variable(name, shape=[3, 1])
        try:
            get_variable(name)
        except ValueError:
            pass
        else:
            self.fail('get_variable should raise when '
                      'trying to create variable already exists.')
